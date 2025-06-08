from math import e
import re
import json
import time
import yaml
from src.engine.config import CONFIG
from src.engine.gost_requests import validate_query
from src.datasets.dataset import Dataset, DatasetFactory
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.utils import execute_sparql_query, get_kgaqa_tracker, get_relative_path, endpoints_fill_parse_args
from src.logging import create_logger, log, LoggingOptions, LogLevel, LogComponent, LogType, logging_fill_parse_args, logging_set_from_args
from tqdm import tqdm
import argparse
import os


names_cache = {}  


# --------------------------------
# ----- EVALUATION FUNCTIONS -----
# --------------------------------

def get_name_query(entity, endpoint_url):
    # print(f"Getting names for entity {entity} from {endpoint_url}")
    
    if not isinstance(entity, str):
        return [(entity,)]
    
    if not entity.startswith("http"):
        return [(entity,)]
    
    if entity in names_cache:
        return names_cache[entity]
    
    def _run_query(query: str):
        try:
            results = execute_sparql_query(query, endpoint_url, False).convert()
        except Exception:
            print("Handle error quietly. Returning empty names.")
            return []

        value_rows = []
        for binding in results.get("results", {}).get("bindings", []):
            row = tuple(v["value"] for v in binding.values())
            value_rows.append(row)
        return sorted(value_rows)

    english_filter = """
            FILTER(LANG(?name) = "" || LANGMATCHES(LANG(?name), "en"))
    """

    if "freebase" in endpoint_url:
        query = f"""
        SELECT ?name WHERE {{
            <{entity}> <http://rdf.freebase.com/ns/type.object.name> ?name .
            {english_filter}
        }}
        """
    elif "wikidata" in endpoint_url:
        query = f"""
        SELECT ?name WHERE {{
            <{entity}> <http://www.w3.org/2000/01/rdf-schema#label> ?name .
            {english_filter}
        }}
        """
    elif "dbpedia" in endpoint_url:
        query = f"""
        SELECT ?name WHERE {{
            <{entity}> <http://www.w3.org/2000/01/rdf-schema#label> ?name .
            {english_filter}
        }}
        """
    elif "beastiary" in endpoint_url or "bestiary" in endpoint_url:
        print(entity)
        entity = entity.split("#")[-1]
        print(entity)
        name = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', entity).lower()
        print(name)
        return [[name]]
    elif "yago2geo" in endpoint_url:
        query = f"""
        SELECT ?name WHERE {{
            VALUES ?prop {{ <http://yago-knowledge.org/resource/hasName>
                            <http://kr.di.uoa.gr/yago2geo/ontology/hasOS_Name>
                            <http://kr.di.uoa.gr/yago2geo/ontology/hasGAG_Name>
                            <http://kr.di.uoa.gr/yago2geo/ontology/hasOSNI_Name>
                            <http://kr.di.uoa.gr/yago2geo/ontology/hasOSI_Name>
                            <http://kr.di.uoa.gr/yago2geo/ontology/hasGADM_Name>
                            <http://kr.di.uoa.gr/yago2geo/ontology/hasOSM_Name>
                          }}
            <{entity}> <http://yago-knowledge.org/resource/hasName> ?name .
            {english_filter}
        }}
        """
    else:
        raise ValueError("Unsupported endpoint URL")
    
    names = _run_query(query)

    # Fallback to any-language label if english-only returned nothing
    if not names and ("wikidata" in endpoint_url or "dbpedia" in endpoint_url or "freebase" in endpoint_url):
        fallback_query = query.replace(english_filter, "")
        names = _run_query(fallback_query)
    
    names_cache[entity] = names
    return names
    
def run_sparql_query_values_only(endpoint_url, query):
    try:
        results = execute_sparql_query(query, endpoint_url).convert()
    except Exception as e:
        log(f"SPARQL query failed for endpoint {endpoint_url}: {e}", LogComponent.KNOWLEDGE_BASE, LogLevel.ERROR, LogType.NORMAL)
        log(f"Query: {query}", LogComponent.KNOWLEDGE_BASE, LogLevel.ERROR, LogType.NORMAL)
        return []

    if "boolean" in results:  # ASK query
        return [[results["boolean"]]]
    
    # SELECT query: extract value tuples
    value_rows = []
    for binding in results["results"]["bindings"]:
        row = tuple(v["value"] for v in binding.values())
        row_with_names = []
        for value in row:
            row_with_names.append(value)
            if value.startswith("http"):
                names = get_name_query(value, endpoint_url)
                if names:
                    row_with_names.append(names[0][0])  # add the first name found
        value_rows.append(tuple(row_with_names))
        # value_rows.append(row)
    return sorted(value_rows)

def get_gold_answer( endpoint_url, gold_query):
    gold = run_sparql_query_values_only(endpoint_url, gold_query)
    gold_columns = [list(row) for row in zip(*gold)]
    
    return gold, gold_columns
    
# -----------------------------
# ----- Logging Functions -----
# -----------------------------
    
import tempfile
    
def atomic_write(filepath, data, mode='w'):
    dir_name = os.path.dirname(filepath)
    with tempfile.NamedTemporaryFile(mode=mode, dir=dir_name, delete=False) as tmp_file:
        json.dump(data, tmp_file, indent=4)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_name = tmp_file.name
    os.replace(temp_name, filepath)  # atomic rename

def save_to_file(run_results, output_file_path):    
    try:
        # Save run results
        atomic_write(output_file_path, run_results)
        print(f"Saved run results to {output_file_path}")

    except Exception as e:
        print(f"Error saving files: {str(e)}")
        
# ----------------
# ----- Main -----
# ----------------

def augment_file(dataset: Dataset, generated_file_path: str):
    
    # -----------------------------------------------------
    # ----- Safeguard from overwriting existing files -----
    # -----------------------------------------------------
    
    generated = {} # map from question to entry. used to evaluate out-of-order questions.
    with open(get_relative_path(generated_file_path), 'r', encoding='utf-8') as f:
        generated_entries = json.load(f)
        for entry in generated_entries:
            generated[entry['question']] = entry
        
    output_file_path = generated_file_path.replace(".json", "_gold_answers.json")
    if os.path.exists(get_relative_path(output_file_path)):
        print(f"Output file {output_file_path} already exists. Please remove it before adding gold answers to prevent overwriting.")
        exit(1)
    
    run_results = []
    
    # ------------------------------
    # ----- Main Functionality -----
    # ------------------------------
    
    KG: KnowledgeGraphs = dataset.get_knowledge_graph()
    
    for idx in range(len(dataset)):     
        print(f"Answering question {idx+1} of {len(dataset)}")
                   
        entry = dataset[idx]
        question = dataset.get_question(entry)

        if question not in generated:
            log(f"Question not found in generated file: {question}", LogComponent.QUERY_GENERATOR, LogLevel.WARNING, LogType.NORMAL)
            continue
        
        gold_query = dataset.get_query(entry)
        gold_answer = None
        # Fix missing prefixes
        fixed_gold_query = ""
        prefixes = dataset.get_prefixes().split("\n")
        for prefix in prefixes:
            prefix = prefix.replace("\n", "").strip()
            if prefix == "":
                continue
            prefix_keyword, prefix_name, prefix_value = prefix.split(" ")
            pattern = r'PREFIX\s+'+re.escape(prefix_name)
            if re.search(pattern, gold_query) is None:
                fixed_gold_query += prefix + "\n"
        fixed_gold_query += gold_query
        gold_query = fixed_gold_query
        
        # If the query can't be answered or is invalid, we generate to have valid results for Gerbil, but we don't count it in the metrics.
        skip_metrics = False
        if validate_query(gold_query) == False:
            log(f"Invalid gold query: {gold_query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
            skip_metrics = True
            continue

        gold_answer, gold_answer_columns = get_gold_answer(KG.endpoint, gold_query)

        generated_entry = generated[question]
        # generated_entry['gold_answer'] = gold_answer
        generated_entry['gold_columns'] = gold_answer_columns

        run_results.append(generated_entry)
        
        log(f"Saving results and metrics after {idx+1} entries...", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
        save_to_file(run_results, output_file_path)
    
    # Final save
    log(f"Final saving results and metrics...", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
    save_to_file(run_results, output_file_path)

if __name__ == "__main__":    
    
    # ----------------------------------
    # ----- Command Line Arguments -----
    # ----------------------------------
    
    parser = argparse.ArgumentParser(
        description="Perform evaluation for generated file."
    )

    parser.add_argument("--generated_file", type=str, required=True, help="File to add gold answers to.")
    
    DatasetFactory.fill_parse_args(parser)
    
    endpoints_fill_parse_args(parser)

    args = parser.parse_args()
    
    # Initialize configuration with parser arguments
    CONFIG(args)
        
    # ------------------------
    # ----- Load Dataset -----
    # ------------------------

    dataset = DatasetFactory.create_from_args(args)
    
    # -------------------------
    # ----- Setup Logging -----
    # -------------------------

    create_logger("SPARQL_EVAL", ".", LoggingOptions.LOG_TO_CONSOLE, LogLevel.INFO)
    
    # -----------------------------------------------------
    # ----- Safeguard from overwriting existing files -----
    # -----------------------------------------------------
    
    if os.path.exists(get_relative_path(args.generated_file)) == False:
        print(f"Generated file {args.generated_file} does not exist.")
        exit(1)
        
    augment_file(dataset, args.generated_file)
