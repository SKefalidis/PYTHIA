from calendar import c
import re
import json
import string
from typing import Tuple, List

from src.engine.config import CONFIG
from src.engine.gost_requests import validate_query, materialize_query
from src.datasets.dataset import DatasetFactory
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.utils import execute_sparql_query, get_kgaqa_tracker, get_relative_path, endpoints_fill_parse_args
from src.logging import create_logger, log, LoggingOptions, LogLevel, LogComponent, LogType, logging_fill_parse_args, logging_set_from_args
import argparse
import os
import csv


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

def compare_queries_loose(question, endpoint_url, csv_answer, gold_query, llm=False, gold_answer=None):
    if gold_answer is None:
        gold_query = materialize_query(gold_query)
        gold = run_sparql_query_values_only(endpoint_url, gold_query)
    else:
        gold = gold_answer

    print(f"CSV Answer Raw: {csv_answer}")
    parsed_answer = csv.reader(csv_answer.strip().splitlines())
    predicted = [tuple(row) for row in parsed_answer]
    print(f"Predicted raw: {predicted}")

    convert_booleans_predicted = []
    for row in predicted:
        new_row = []
        for value in row:
            if value.lower() == "true":
                new_row.append(True)
                new_row.append('true')
            elif value.lower() == "false":
                new_row.append(False)
                new_row.append('false')
            else:
                new_row.append(value.lower().strip())
        convert_booleans_predicted.append(tuple(new_row))
    predicted = convert_booleans_predicted
    
    if not predicted or not gold:
        return 0, len(predicted), len(gold), 0, 0, 0, 0
    
    predicted_columns = [list(row) for row in zip(*predicted)]
    if gold_answer is None:
        gold_columns = [list(row) for row in zip(*gold)]
    else:
        gold_columns = [gold]
    for column in gold_columns:
        for i in range(len(column)):
            if isinstance(column[i], str):
                column[i] = column[i].lower()

    print(f"Predicted: {predicted_columns}")
    print(f"Gold: {gold_columns}")
    
    # print(f"Result 1: {result1}")
    # print(f"Result 2: {result2}")
    # print(f"Result 1 columns: {result1_columns}")
    # print(f"Result 2 columns: {result2_columns}")
    
    best_tp, best_fp, best_fn = 0, 0, 0
    for i in predicted_columns:
        print(f"Evaluating predicted column: {i}")
        for j in gold_columns:
            print(f"Against gold column: {j}")
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
                break

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

def tog_match(results: str, gold_query: str, endpoint_url: str) -> Tuple[int, int, int]:
    "Returns (formatted_properly, exact_match, exact_match_extract)"
    def check_string(string):
        return "{" in string

    def clean_results(string):
        if "{" in string:
            start = string.find("{") + 1
            end = string.find("}")
            content = string[start:end]
            return content
        else:
            return "NULL"
        
    def check_refuse(string):
        refuse_words = ["however", "sorry"]
        return any(word in string.lower() for word in refuse_words)

    def exact_match(response, answers):
        clean_result = response.strip().replace(" ","").lower()
        for answer in answers:
            clean_answer = answer.strip().replace(" ","").lower()
            if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
                return True
        return False

    def extract_content(s):
        matches = re.findall(r'\{(.*?)\}', s)
        if len(matches) >= 2 and matches[0].lower() == 'yes':
            return matches[1]
        elif len(matches) >= 1:
            return matches[0]
        else:
            return 'NULL'
        
    if gold_answer is None:
        gold_query = materialize_query(gold_query)
        gold = run_sparql_query_values_only(endpoint_url, gold_query)
    else:
        gold = gold_answer

    if gold_answer is None:
        gold_columns = [list(row) for row in zip(*gold)]
    else:
        gold_columns = [gold]
    for column in gold_columns:
        for i in range(len(column)):
            if isinstance(column[i], str):
                column[i] = column[i].lower()

    answers = [item.lower() if isinstance(item, str) else str(item).lower() for gold_column in gold_columns for item in gold_column]
    
    match = 0
    match_extract = 0
    formatted_properly = 0
    if check_string(results):
        response = clean_results(results)
        if response=="NULL":
            response = results
        else:
            if exact_match(response, answers):
                match = 1
                formatted_properly = 1
        
        response = extract_content(results)
        if response=="NULL":
            response = results
        else:
            if exact_match(response, answers):
                match_extract = 1
                formatted_properly = 1
    else:
        response = results
        if exact_match(response, answers):
            match = 1
            match_extract = 1
        
    return formatted_properly, match, match_extract

def pog_match(results: str, gold_query: str, endpoint_url: str) -> int:
    "Returns (formatted_properly, exact_match, exact_match_extract)"

    def clean_results(string):
        # top_list_str = ""
        # match = re.search(r'swer:\s*\{([^}]+)\}', string)
        # if not match:
        #     # match = re.search(r'list:\s*\{([^}]+)\}', text)
        #     return ""
        
        # top_list_str = match.group(1)
        # return top_list_str
    
    # Adjust the function to handle a single string input

        # Split the input string by 'answer:' to isolate each section
        # print("++++++++++++++++++++")
        # print(string)
        if "answer:{" not in string:
            return []
        # else:   
            # print("+++==========++++++++++")
        sections = string.split('answer:')
        
        all_answers = []
        
        for section in sections[1:]:  # Skip the first part since it doesn't contain an answer
            # Extract the part between curly braces
            # print(section)
            # print("++++++++++++++++++++")
            # print(string)
            #get string after "answer:" in one line from section
            string = section.split("\n")[0]
            replace_string = string.replace("{",",").replace("}",",")
            # answers = section.split('{')[1].split('}')[0]
            # Split by comma and strip any spaces
            answers_list = [answer.strip() for answer in replace_string.split(',')]
            # Add to the overall list
            all_answers.extend(answers_list)
        
        # Remove duplicates and return the final list
        all_answers = list(set(all_answers))
        all_answers = [x for x in all_answers if x != ""]
        return list(set(all_answers))

    def check_answer(answer: str, answer_list):
        lower_answer = answer.strip().replace(" ","").lower()
        # print(f"LLM Answer Processed: {lower_answer}")
        # print("### answer ends ##")
        getanswer = clean_results(lower_answer)
        # print(f"Extracted Answers: {getanswer}")
        for answer_name in answer_list:
            lower_answer_name = answer_name.strip().replace(" ","").lower()
            if lower_answer_name in lower_answer:
                # print("answer is found in the LLM answer")
                return True

        if len(getanswer) > 0:
            for getanswer_e in getanswer:
                for answer_name in answer_list:
                    lower_answer_name = answer_name.strip().replace(" ","").lower()
                    if getanswer_e in lower_answer_name:
                        # print("answer is found in the LLM answer 2")
                        return True
        return False
        
    if gold_answer is None:
        gold_query = materialize_query(gold_query)
        gold = run_sparql_query_values_only(endpoint_url, gold_query)
    else:
        gold = gold_answer

    if gold_answer is None:
        gold_columns = [list(row) for row in zip(*gold)]
    else:
        gold_columns = [gold]
    for column in gold_columns:
        for i in range(len(column)):
            if isinstance(column[i], str):
                column[i] = column[i].lower()

    answers = [item.lower() if isinstance(item, str) else str(item).lower() for gold_column in gold_columns for item in gold_column]
    
    match = check_answer(results, answers)
        
    return 1 if match else 0

def compute_metrics(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}

def query_has_results(endpoint_url, query, length = 0):
    try:
        query = materialize_query(query)
        results = execute_sparql_query(query, endpoint_url).convert()
        if "boolean" in results:
            return results["boolean"] # Although this skips ASK queries that return False, it avoids counting queries that have missing triples in our version of the KG.
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

def save_to_file(run_results, metrics):    
    try:
        # Save run results
        atomic_write(output_file_path, run_results)
        print(f"Saved run results to {output_file_path}")

        # Save metrics
        atomic_write(outputs_metrics_path, metrics)
        print(f"Saved metrics to {outputs_metrics_path}")

    except Exception as e:
        print(f"Error saving files: {str(e)}")


if __name__ == "__main__":    
    
    # ----------------------------------
    # ----- Command Line Arguments -----
    # ----------------------------------
    
    parser = argparse.ArgumentParser(
        description="Perform evaluation for generated file."
    )

    parser.add_argument("--generated_file", type=str, required=True, help="Path to the answer file containing answers in CSV format.")
    parser.add_argument("--system", type=str, required=True, choices=["inherent", "tog", "rog", "pog"], help="The system used to generate the answers.")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing output files.")
    parser.add_argument("--end_idx", type=int, default=None, help="Limit evaluation to the first N questions (exclusive).")
    
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

    create_logger("TEXT_EVAL", ".", LoggingOptions.LOG_TO_CONSOLE, LogLevel.INFO)
    
    # -----------------------------------------------------
    # ----- Safeguard from overwriting existing files -----
    # -----------------------------------------------------
    
    if os.path.exists(get_relative_path(args.generated_file)) == False:
        print(f"Generated file {args.generated_file} does not exist.")
        exit(1)
        
    generated = {} # map from question to entry. used to evaluate out-of-order questions.
    with open(get_relative_path(args.generated_file), 'r', encoding='utf-8') as f:
        generated_entries = json.load(f)
        for entry in generated_entries:
            generated[entry['question']] = entry
        
    output_file_path = args.generated_file.replace(".json", "_eval.json")
    if os.path.exists(get_relative_path(output_file_path)) and not args.overwrite:
        print(f"Output file {output_file_path} already exists. Please remove it before running evaluation to prevent overwriting.")
        exit(1)
        
    outputs_metrics_path = args.generated_file.replace(".json", "_eval_metrics.json")
    if os.path.exists(get_relative_path(outputs_metrics_path)) and not args.overwrite:
        print(f"Metrics file {outputs_metrics_path} already exists. Please remove it before running evaluation to prevent overwriting.")
        exit(1)
    
    # will hold the metrics. less than our own engine since we only do SPARQL evaluation here.    
    metrics = {
        # dataset metrics
        "total": 0,
        "total_valid": 0,
        "missing_questions": 0,
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
        "total_pog_match": 0,
        "total_tog_em": 0,
        "total_tog_em_fixed": 0,
        "total_tog_formatted_properly": 0,
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
        "pog_match": 0,
        "tog_em": 0,
        "tog_em_fixed": 0,
        "tog_formatted_properly": 0,
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
    
    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    
    for idx in range(end_idx):     
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
            
        metrics["total_valid"] += 1
        
        if question not in generated:
            if question[:-1] in generated:
                question = question[:-1]
            else:
                log(f"Question not found in generated answers: {question}", LogComponent.QUERY_GENERATOR, LogLevel.ERROR, LogType.NORMAL)
                metrics["missing_questions"] += 1
                continue
        
        generated_entry = generated[question]

        if args.system == "rog": # RoG
            csv_answer = generated_entry['prediction']
        elif args.system == "tog" or args.system == "pog": # ToG & PoG
            text_answer = generated_entry['results']
            csv_answer = generated_entry['csv_answer']
        elif args.system == "inherent": # LLM inherent
            answer = generated_entry['answer']
            if isinstance(answer, list) and all(isinstance(row, list) for row in answer):
                output = []
                for row in answer:
                    output.append(','.join([str(item) for item in row]))
                csv_answer = '\n'.join(output)
            else:
                print(f"Unsupported answer format for question: {question}")
                exit(1)
        else:
            print(f"No answer found in generated entry for question: {question}")
            exit(1)
        
        generated_metrics = generated_entry.get('metrics', {})
        try:
            metrics['total_time'] += generated_entry.get('elapsed', 0)
            if generated_metrics['SPARQL_CALLS'] != 'Unknown':
                metrics['total_sparql_calls'] += generated_metrics.get('SPARQL_CALLS', 0)
                metrics['total_sparql_time'] += generated_metrics.get('SPARQL_TIME', 0)
                metrics['total_llm_calls'] += generated_metrics.get('LLM_CALLS', 0)
                metrics['total_llm_time'] += generated_metrics.get('LLM_TIME', 0)
                metrics['total_llm_inputs'] += generated_metrics.get('LLM_INPUTS', 0)
                metrics['total_llm_outputs'] += generated_metrics.get('LLM_OUTPUTS', 0)
        except Exception as e:
            print(metrics)
            print(generated_metrics)
            print(f"Error updating metrics for question: {question}, error: {e}")
            raise e
        
        # --------------------------
        # ----- Update Metrics -----
        # --------------------------
        
        if skip_metrics == False:
            # Zeroshot
            log(f"Gold query: {gold_query}", LogComponent.OTHER, LogLevel.APPLICATION, LogType.GOLD)
            log(f"Generated query: {generated_entry}", LogComponent.OTHER, LogLevel.APPLICATION, LogType.NORMAL)

            tp, fp, fn, hits_at_1, hits, lax_hits, laxxer_hits = compare_queries_loose(question, KG.endpoint, csv_answer, gold_query, gold_answer=gold_answer)

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

            tog_formatted_properly, tog_em, to_em_fixed = tog_match(text_answer, gold_query, KG.endpoint)

            metrics["total_tog_em"] += tog_em
            metrics["total_tog_em_fixed"] += to_em_fixed
            metrics["total_tog_formatted_properly"] += int(tog_formatted_properly)

            pog_match_result = pog_match(text_answer, gold_query, KG.endpoint)
            metrics["total_pog_match"] += pog_match_result

            # if pog_match_result == 1 and laxxer_hits == 0:
            #     print("+++++++++++++")
            #     print("Pog match is 1 but laxxer hits is 0")
            #     print("Gold query:")
            #     print(gold_query)
            #     print("CSV answer:")
            #     print(csv_answer)
            #     print("Generated answer:")
            #     print(text_answer)
            #     print("+++++++++++++")
            #     if idx + 1 != 40:
            #         exit(1)

        else:
            tp, fp, fn, hits_at_1 = "SKIPPED", "SKIPPED", "SKIPPED", "SKIPPED"
            hits, lax_hits, laxxer_hits = "SKIPPED", "SKIPPED", "SKIPPED"
            tog_em, to_em_fixed, tog_formatted_properly = "SKIPPED", "SKIPPED", "SKIPPED"
            pog_match_result = "SKIPPED"
            
            
        entry = {
            "question": question,
            "csv_answer": csv_answer,
            "results" : {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "exact_match": tp > 0 and fp == 0 and fn == 0 if skip_metrics == False else "SKIPPED",
                "hits_at_1": hits_at_1,
                "hits": hits if skip_metrics == False else "SKIPPED",
                "lax_hits": lax_hits if skip_metrics == False else "SKIPPED",
                "laxxer_hits": laxxer_hits if skip_metrics == False else "SKIPPED",
                "pog_match": pog_match_result,
                "tog_em": tog_em,
                "tog_em_fixed": to_em_fixed,
                "tog_formatted_properly": tog_formatted_properly
            },
            "metrics": generated_metrics
        }
        
        run_results.append(entry)
        
        metrics["exact_match"] = metrics["total_exact_match"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["f1"] = metrics["total_macro_f1"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["precision"] = metrics["total_macro_precision"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["recall"] = metrics["total_macro_recall"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["hits_at_1"] = metrics["total_hits_at_1"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["hits"] = metrics["total_hits"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["lax_hits"] = metrics["total_lax_hits"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["laxxer_hits"] = metrics["total_laxxer_hits"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["pog_match"] = metrics["total_pog_match"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["tog_em"] = metrics["total_tog_em"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["tog_em_fixed"] = metrics["total_tog_em_fixed"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["tog_formatted_properly"] = metrics["total_tog_formatted_properly"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["average_time_per_question"] = metrics["total_time"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["average_sparql_calls_per_question"] = metrics["total_sparql_calls"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["average_sparql_time_per_question"] = metrics["total_sparql_time"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["average_llm_calls_per_question"] = metrics["total_llm_calls"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["average_llm_time_per_question"] = metrics["total_llm_time"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["average_llm_inputs_per_question"] = metrics["total_llm_inputs"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        metrics["average_llm_outputs_per_question"] = metrics["total_llm_outputs"] / metrics["total_valid"] if metrics["total_valid"] > 0 else 0
        
        log(f"Saving results and metrics after {idx+1} entries...", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
        save_to_file(run_results, metrics)
    
    # Final save
    log(f"Final saving results and metrics...", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
    save_to_file(run_results, metrics)
