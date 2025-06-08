from math import e
import re
import json
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


# --------------------------------
# ----- EVALUATION FUNCTIONS -----
# --------------------------------
    
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
    # print(results)
    for binding in results["results"]["bindings"]:
        row = tuple(v["value"] for v in binding.values())
        value_rows.append(row)
        
    # print(value_rows)
    # print(query)

    # Add labels for URIs
    if DISABLE_LABELS == False:
        if len(value_rows) < 100: # to avoid excessively slow evaluation
            new_value_rows = []
            for row in value_rows:
                row_with_names = []
                for value in row:
                    row_with_names.append(value)
                    if value.startswith("http"):
                        names = get_name_query(value, endpoint_url)
                        if names:
                            try:
                                row_with_names.append(names[0][0])  # add the first name found
                            except Exception as e:
                                print(names)
                                print(e)
                                raise e
                new_value_rows.append(tuple(row_with_names))
            value_rows = new_value_rows
        
    max_row_len = max(len(row) for row in value_rows) if value_rows else 0
    padded_value_rows = []
    for row in value_rows:
        padded_row = list(row) + ["" for _ in range(max_row_len - len(row))]
        padded_value_rows.append(tuple(padded_row))

    return sorted(padded_value_rows)

def compare_queries_loose(endpoint_url, query, gold_query, answer=None, gold_answer=None):
    if answer is None:
        predicted = run_sparql_query_values_only(endpoint_url, query)
    else:
        predicted = answer

    if gold_answer is None:
        gold = run_sparql_query_values_only(endpoint_url, gold_query)
    else:
        gold = gold_answer
    
    if predicted == [] or gold == []:
        return 0, len(predicted), len(gold), 0, 0, 0, 0
    
    predicted_columns = [list(row) for row in zip(*predicted)]
    if gold_answer is None:
        gold_columns = [list(row) for row in zip(*gold)]
    else:
        gold_columns = [gold]
    
    # print(f"Predicted: {predicted_columns}")
    # print(f"Gold: {gold_columns}")
    
    # print(f"Result 1: {result1}")
    # print(f"Result 2: {result2}")
    # print(f"Result 1 columns: {result1_columns}")
    # print(f"Result 2 columns: {result2_columns}")
    
    best_tp, best_fp, best_fn = 0, 0, 0
    for i in predicted_columns:
        for j in gold_columns:
            tp = len(set(i) & set(j))
            fp = len(set(i) - set(j))
            fn = len(set(j) - set(i))
            if tp > best_tp: # FIXME: bug, does not count fp/fn correctly if tp is always 0, but it does not affect the metrics, so we ignore it for now
                best_tp, best_fp, best_fn = tp, fp, fn

    hits_at_1 = 0
    for i in predicted[0]:
        for j in gold_columns:
            if i in j:
                hits_at_1 = 1

    flat_predictions_list = [item.lower() if isinstance(item, str) else item for pred_column in predicted_columns for item in pred_column]
    flat_gold_list = [item.lower() if isinstance(item, str) else item for gold_column in gold_columns for item in gold_column]

    hits = 0
    lax_hits = 0
    laxxer_hits = 0

    for g in flat_gold_list:
        if g in flat_predictions_list:
            hits = 1
            lax_hits = 1
            laxxer_hits = 1
            break
        else:
            for p in flat_predictions_list:
                if not isinstance(p, bool) and not isinstance(g, bool):
                    if g in p:
                        lax_hits = 1
                        laxxer_hits = 1
                        break
                
                if not isinstance(g, bool) and not isinstance(p, bool):
                    if p in g:
                        laxxer_hits = 1
                        break
            if lax_hits > 0:
                break
    
    return best_tp, best_fp, best_fn, hits_at_1, hits, lax_hits, laxxer_hits

def compute_metrics(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}

def query_has_results(endpoint_url, query, length = 0):
    try:
        results = execute_sparql_query(query, endpoint_url).convert()
        if "boolean" in results:
            return True #results["boolean"]
        else:
            has_anything = len(results["results"]["bindings"]) > 0
            # if length > 0:
            #     has_less_than = len(results["results"]["bindings"]) < length
            # else:
            #     has_less_than = True
            return has_anything # and has_less_than
    except Exception as e:
        # print(f"SPARQL query failed for endpoint {endpoint_url}: {e}")
        log(f"SPARQL query failed for endpoint {endpoint_url}: {e}", LogComponent.KNOWLEDGE_BASE, LogLevel.ERROR, LogType.NORMAL)
        log(f"Query: {query}", LogComponent.KNOWLEDGE_BASE, LogLevel.ERROR, LogType.NORMAL)
        return False
    
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

def save_to_file(run_results, metrics, output_file_path, outputs_metrics_path):    
    try:
        # Save run results
        atomic_write(output_file_path, run_results)
        print(f"Saved run results to {output_file_path}")

        # Save metrics
        atomic_write(outputs_metrics_path, metrics)
        print(f"Saved metrics to {outputs_metrics_path}")

    except Exception as e:
        print(f"Error saving files: {str(e)}")
        
# ----------------
# ----- Main -----
# ----------------

def evaluate_generated_file(dataset: Dataset, generated_file_path: str, end_idx: int = -1):
    
    # -----------------------------------------------------
    # ----- Safeguard from overwriting existing files -----
    # -----------------------------------------------------
    
    generated = {} # map from question to entry. used to evaluate out-of-order questions.
    with open(get_relative_path(generated_file_path), 'r', encoding='utf-8') as f:
        generated_entries = json.load(f)
        if end_idx == -1:
            end_idx = len(generated_entries)
        for entry in generated_entries[:end_idx]:
            generated[entry['question']] = entry
        
    disable_labels_str = "_no_labels" if args.disable_labels else ""
    output_file_path = generated_file_path.replace(".json", f"_eval{disable_labels_str}.json")
    if os.path.exists(get_relative_path(output_file_path)):
        print(f"Output file {output_file_path} already exists. Please remove it before running evaluation to prevent overwriting.")
        return
        
    outputs_metrics_path = generated_file_path.replace(".json", f"_eval_metrics{disable_labels_str}.json")
    if os.path.exists(get_relative_path(outputs_metrics_path)):
        print(f"Metrics file {outputs_metrics_path} already exists. Please remove it before running evaluation to prevent overwriting.")
        return
    
    # will hold the metrics. less than our own engine since we only do SPARQL evaluation here.    
    metrics = {
        # dataset metrics
        "total": 0,
        "total_valid": 0,
        "invalid_gold_queries": 0,
        "empty_gold_queries": 0,
        # accuracy metrics
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0,
        "total_exact_match": 0,
        "total_macro_f1": 0,
        "total_macro_precision": 0,
        "total_macro_recall": 0,
        "total_hits_at_1": 0,
        "total_hits": 0,
        "total_lax_hits": 0,
        "total_laxxer_hits": 0,
        # efficiency metrics
        "total_time": 0,
        "total_sparql_calls": 0,
        "total_sparql_time": 0,
        "total_llm_calls": 0,
        "total_llm_time": 0,
        "total_llm_inputs": 0,
        "total_llm_outputs": 0,
        # average metrics to be computed at the end
        "exact_match": 0,
        "f1": 0,
        "precision": 0,
        "recall": 0,
        "hits_at_1": 0,
        "hits": 0,
        "lax_hits": 0,
        "laxxer_hits": 0,
        "average_time_per_question": 0,
        "average_sparql_calls_per_question": 0,
        "average_sparql_time_per_question": 0,
        "average_llm_calls_per_question": 0,
        "average_llm_time_per_question": 0,
        "average_llm_inputs_per_question": 0,
        "average_llm_outputs_per_question": 0,
    }
    
    run_results = []
    
    # ------------------------------
    # ----- Main Functionality -----
    # ------------------------------
    
    KG: KnowledgeGraphs = dataset.get_knowledge_graph()
    
    for idx in range(len(dataset)):     
        if idx == end_idx:
            break
        print(f"Answering question {idx+1} of {len(dataset)}")
                   
        entry = dataset[idx]
        question = dataset.get_question(entry)
        metrics['total'] += 1
        
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
            metrics["invalid_gold_queries"] += 1
            skip_metrics = True
            continue

        if query_has_results(KG.endpoint, gold_query) == False:
            log(f"Empty gold query: {gold_query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
            metrics["empty_gold_queries"] += 1
            skip_metrics = True
            continue
        
        if question in generated:
            metrics["total_valid"] += 1
            generated_entry = generated[question]
            if 'sparql' in generated_entry:
                query = generated_entry['sparql']
            else:
                query = None

            if 'answer' in generated_entry:
                answer = generated_entry['answer']
            else:
                answer = None

            generated_metrics = generated_entry.get('metrics', {})
            metrics['total_time'] += generated_entry.get('elapsed', 0)
            metrics['total_sparql_calls'] += generated_metrics.get('SPARQL_CALLS', 0)
            metrics['total_sparql_time'] += generated_metrics.get('SPARQL_TIME', 0)
            metrics['total_llm_calls'] += generated_metrics.get('LLM_CALLS', 0)
            metrics['total_llm_time'] += generated_metrics.get('LLM_TIME', 0)
            metrics['total_llm_inputs'] += generated_metrics.get('LLM_INPUTS', 0)
            metrics['total_llm_outputs'] += generated_metrics.get('LLM_OUTPUTS', 0)
        else:
            query = ""
            answer = None
            generated_metrics = {}
        
        # --------------------------
        # ----- Update Metrics -----
        # --------------------------

        if skip_metrics == False:
            # Zeroshot
            log(f"Gold query: {gold_query}", LogComponent.OTHER, LogLevel.APPLICATION, LogType.GOLD)
            log(f"Generated query: {query}", LogComponent.OTHER, LogLevel.APPLICATION, LogType.NORMAL)

            tp, fp, fn, hits_at_1, hits, lax_hits, laxxer_hits = compare_queries_loose(KG.endpoint, query, gold_query, answer=answer, gold_answer=gold_answer)

            if tp > 0 and fp == 0 and fn == 0:
                log(f"✅ Correct: {question}", LogComponent.OTHER, LogLevel.APPLICATION, LogType.NORMAL)
                metrics['total_exact_match'] += 1
            else:
                log(f"❌ Incorrect: {question}", LogComponent.OTHER, LogLevel.APPLICATION, LogType.NORMAL)
                
            log(f"{tp} TP, {fp} FP, {fn} FN", LogComponent.OTHER, LogLevel.APPLICATION, LogType.NORMAL)
            log(f"Hits@1: {hits_at_1}", LogComponent.OTHER, LogLevel.APPLICATION, LogType.NORMAL)
            log(f"Hits (strict): {hits}", LogComponent.OTHER, LogLevel.APPLICATION, LogType.NORMAL)
            log(f"Hits (lax): {lax_hits}", LogComponent.OTHER, LogLevel.APPLICATION, LogType.NORMAL)
            log(f"Hits (laxxer): {laxxer_hits}", LogComponent.OTHER, LogLevel.APPLICATION, LogType.NORMAL)

            metrics["total_tp"] += tp
            metrics["total_fp"] += fp
            metrics["total_fn"] += fn
            metrics["total_hits_at_1"] += hits_at_1
            metrics["total_hits"] = metrics.get("total_hits", 0) + hits
            metrics["total_lax_hits"] = metrics.get("total_lax_hits", 0) + lax_hits
            metrics["total_laxxer_hits"] = metrics.get("total_laxxer_hits", 0) + laxxer_hits
            
            # Compute metrics
            metrics_internal = compute_metrics(tp, fp, fn)            
            metrics["total_macro_f1"] += metrics_internal['f1']
            metrics["total_macro_precision"] += metrics_internal['precision']
            metrics["total_macro_recall"] += metrics_internal['recall']
            
            log(f"\nMacro-averaged metrics: {metrics_internal}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
            log(f"\tPrecision: {metrics['total_macro_precision']/metrics['total_valid']:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
            log(f"\tRecall: {metrics['total_macro_recall']/metrics['total_valid']:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
            log(f"\tF1: {metrics['total_macro_f1']/metrics['total_valid']:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
            log(f"\tHits@1: {metrics['total_hits_at_1']/metrics['total_valid']:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)

        else:
            tp, fp, fn, hits_at_1 = "SKIPPED", "SKIPPED", "SKIPPED", "SKIPPED"
            
            
        entry = {
            "question": question,
            "generated_query": query,
            "generated_answer": answer,
            "results" : {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "hits_at_1": hits_at_1,
                "hits": hits,
                "lax_hits": lax_hits,
                "laxxer_hits": laxxer_hits,
                "exact_match": tp > 0 and fp == 0 and fn == 0 if skip_metrics == False else "SKIPPED"
            },
            "metrics": generated_metrics
        }
        
        run_results.append(entry)
        
        metrics["exact_match"] = metrics["total_exact_match"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["hits_at_1"] = metrics["total_hits_at_1"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["hits"] = metrics["total_hits"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["lax_hits"] = metrics["total_lax_hits"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["laxxer_hits"] = metrics["total_laxxer_hits"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["f1"] = metrics["total_macro_f1"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["precision"] = metrics["total_macro_precision"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["recall"] = metrics["total_macro_recall"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["average_time_per_question"] = metrics["total_time"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["average_sparql_calls_per_question"] = metrics["total_sparql_calls"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["average_sparql_time_per_question"] = metrics["total_sparql_time"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["average_llm_calls_per_question"] = metrics["total_llm_calls"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["average_llm_time_per_question"] = metrics["total_llm_time"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["average_llm_inputs_per_question"] = metrics["total_llm_inputs"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["average_llm_outputs_per_question"] = metrics["total_llm_outputs"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        
        log(f"Saving results and metrics after {idx+1} entries...", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
        save_to_file(run_results, metrics, output_file_path, outputs_metrics_path)
    
    # Final save
    log(f"Final saving results and metrics...", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
    save_to_file(run_results, metrics, output_file_path, outputs_metrics_path)

if __name__ == "__main__":    
    
    # ----------------------------------
    # ----- Command Line Arguments -----
    # ----------------------------------
    
    parser = argparse.ArgumentParser(
        description="Perform evaluation for generated file."
    )

    parser.add_argument("--generated_file", type=str, required=True, help="Path to the generated file containing SPARQL queries and/or answers.")
    parser.add_argument("--file_name", type=str, required=False, help="Used for directory input.")
    parser.add_argument("--end_idx", type=int, default=-1, help="Used to limit number of questions processed.")
    parser.add_argument("--disable_labels", action="store_true", help="Disable fetching labels for URIs in SPARQL results. This can speed up evaluation significantly, but results will be less interpretable.")
    
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
    
    # ----------------
    # ----- Main -----
    # ----------------
    
    DISABLE_LABELS = args.disable_labels

    if "," in args.generated_file or (os.path.exists(args.generated_file) and os.path.isfile(args.generated_file)):
        generated_files = args.generated_file.split(",")
        for generated_file in generated_files:
            if os.path.exists(generated_file) == False:
                print(f"Generated file {generated_file} does not exist.")
                continue

            evaluate_generated_file(dataset, generated_file, args.end_idx)
    elif os.path.exists(args.generated_file) and os.path.isdir(args.generated_file):
        target_names = [args.file_name] if args.file_name else ""
        if target_names == "":
            target_names = [args.dataset + "_generated.json"]
            target_names.append(args.dataset + "_fewshot_generated.json")
            target_names.append(args.dataset + "_cot_generated.json")
        print(f"Searching directory for generated files \"{target_names}\"...")
        for root, dirs, files in os.walk(args.generated_file):
            for filename in files:
                for target_name in target_names:
                    if filename.endswith(target_name):
                        generated_file = os.path.join(root, filename)
                        evaluate_generated_file(dataset, generated_file, args.end_idx)
    else:
        print(f"Generated file path {args.generated_file} is not a valid file, file list or directory.")
