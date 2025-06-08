import os
import sys
import json
import argparse
import tempfile
from typing import Any, Dict, List

import requests
from scipy.special import entr
from tqdm import tqdm

from src.datasets.dataset import DatasetFactory
from src.utils import execute_sparql_query

from src.logging import create_logger, create_console_logger

def atomic_write(filepath: str, data: Any, mode: str = "w") -> None:
    """Atomically write JSON data to filepath."""
    dir_name = os.path.dirname(filepath)
    os.makedirs(dir_name, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode=mode, dir=dir_name, delete=False) as tmp_file:
        json.dump(data, tmp_file, indent=4)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_name = tmp_file.name
    os.replace(temp_name, filepath)


def call_grasp_api(
    server_url: str,
    question: str,
    kg: str,
    timeout: int = 180,
    topic_entities: str | None = None,
    examples: str | None = None,
) -> Dict[str, Any]:

    payload = {
        "task": "sparql-qa", 
        "input": question, 
        "knowledge_graphs": [kg],
        "topic_entities": topic_entities,
        "examples": examples,
    }
    response = requests.post(
        f"{server_url}/run",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def extract_sparql(resp: Dict[str, Any]) -> str | None:
    if isinstance(resp, dict) and isinstance(resp.get("sparql"), str):
        return resp["sparql"]
    output = resp.get("output") if isinstance(resp, dict) else None
    if isinstance(output, dict) and isinstance(output.get("sparql"), str):
        return output["sparql"]
    return None


def extract_metrics(resp: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(resp, dict):
        return {}
    if isinstance(resp.get("metrics"), dict):
        return resp["metrics"]
    output = resp.get("output")
    if isinstance(output, dict) and isinstance(output.get("metrics"), dict):
        return output["metrics"]
    return {}


def run_sparql_query_values_only(endpoint_url: str, query: str | None) -> List[tuple]:
    if not query:
        return []
    try:
        results = execute_sparql_query(query, endpoint_url).convert()
    except Exception as e:
        print("Error executing SPARQL query:", e)
        return []

    if "boolean" in results:
        return [[results["boolean"]]]

    value_rows = []
    for binding in results.get("results", {}).get("bindings", []):
        row = tuple(v.get("value") for v in binding.values())
        value_rows.append(row)
    return sorted(value_rows)


def compare_queries_loose(endpoint_url: str, query: str | None, gold_query: str | None) -> tuple[int, int, int, int]:
    predicted = run_sparql_query_values_only(endpoint_url, query)
    gold = run_sparql_query_values_only(endpoint_url, gold_query)

    print("Predicted Results:", predicted)
    print("Gold Results:", gold)

    if not predicted or not gold:
        return 0, len(predicted), len(gold), 0

    predicted_columns = [list(row) for row in zip(*predicted)]
    gold_columns = [list(row) for row in zip(*gold)]

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

    print(f"Comparison - TP: {best_tp}, FP: {best_fp}, FN: {best_fn}, Hits@1: {hits_at_1}")

    return best_tp, best_fp, best_fn, hits_at_1


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def merge_metrics(total: Dict[str, float], metrics: Dict[str, Any]) -> None:
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            total[key] = total.get(key, 0.0) + float(val)


def compute_eval_summary(eval_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": len(eval_entries),
        "total_valid": 0,
        "invalid_gold_queries": 0,
        "empty_gold_queries": 0,
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0,
        "total_exact_match": 0,
        "total_hits_at_1": 0,
        "total_macro_f1": 0,
        "total_macro_precision": 0,
        "total_macro_recall": 0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "total_time": 0,
        "total_sparql_calls": 0,
        "total_sparql_time": 0,
        "total_llm_calls": 0,
        "total_llm_time": 0,
        "total_llm_inputs": 0,
        "total_llm_outputs": 0,
    }

    for item in eval_entries:
        res = item.get("results", {}) if isinstance(item, dict) else {}
        metrics = item.get("metrics", {}) if isinstance(item, dict) else {}

        gold_query = item.get("gold_query") if isinstance(item, dict) else None
        if gold_query:
            summary["total_valid"] += 1
        else:
            summary["empty_gold_queries"] += 1

        for key, target in [("tp", "total_tp"), ("fp", "total_fp"), ("fn", "total_fn")]:
            val = res.get(key)
            if isinstance(val, (int, float)):
                summary[target] += val

        if res.get("exact_match"):
            summary["total_exact_match"] += 1
        if res.get("hits_at_1"):
            summary["total_hits_at_1"] += 1

        if isinstance(res.get("f1"), (int, float)):
            summary["total_macro_f1"] += res["f1"]
        if isinstance(res.get("precision"), (int, float)):
            summary["total_macro_precision"] += res["precision"]
        if isinstance(res.get("recall"), (int, float)):
            summary["total_macro_recall"] += res["recall"]

        for k, target in [
            ("elapsed", "total_time"),
            ("SPARQL_CALLS", "total_sparql_calls"),
            ("SPARQL_TIME", "total_sparql_time"),
            ("LLM_CALLS", "total_llm_calls"),
            ("LLM_TIME", "total_llm_time"),
            ("LLM_INPUTS", "total_llm_inputs"),
            ("LLM_OUTPUTS", "total_llm_outputs"),
        ]:
            val = metrics.get(k)
            if isinstance(val, (int, float)):
                summary[target] += float(val)

    total_valid = max(summary.get("total_valid", 0), 1)
    summary["hits_at_1"] = summary["total_hits_at_1"] / total_valid
    summary["exact_match"] = summary["total_exact_match"] / total_valid

    def safe_ratio(num: float, den: float) -> float:
        return num / den if den else 0.0

    macro_den = summary.get("total_valid", 0)
    if macro_den:
        summary["precision"] = summary["total_macro_precision"] / macro_den
        summary["recall"] = summary["total_macro_recall"] / macro_den
        summary["f1"] = summary["total_macro_f1"] / macro_den
    else:
        summary["precision"] = 0.0
        summary["recall"] = 0.0
        summary["f1"] = 0.0

    total_questions = max(summary.get("total", 0), 1)
    summary["average_time_per_question"] = safe_ratio(summary["total_time"], total_questions)
    summary["average_sparql_calls_per_question"] = safe_ratio(summary["total_sparql_calls"], total_questions)
    summary["average_sparql_time_per_question"] = safe_ratio(summary["total_sparql_time"], total_questions)
    summary["average_llm_calls_per_question"] = safe_ratio(summary["total_llm_calls"], total_questions)
    summary["average_llm_time_per_question"] = safe_ratio(summary["total_llm_time"], total_questions)
    summary["average_llm_inputs_per_question"] = safe_ratio(summary["total_llm_inputs"], total_questions)
    summary["average_llm_outputs_per_question"] = safe_ratio(summary["total_llm_outputs"], total_questions)

    return summary


def persist_all(
    base_path: str,
    results: List[Dict[str, Any]],
    results_full: List[Dict[str, Any]],
    metrics_summary: Dict[str, Any],
) -> None:
    save_targets = [
        (results, base_path + "_results.json"),
        (results_full, base_path + "_results_full.json"),
        (metrics_summary, base_path + "_metrics.json"),
        # (eval_results, base_path + "_results_eval.json"),
        # (eval_metrics_summary, base_path + "_results_eval_metrics.json"),
    ]
    for data, path in save_targets:
        atomic_write(path, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GRASP text-to-SPARQL over a dataset and save outputs.")
    parser.add_argument("--dataset", type=str, help="Dataset name for retrieving examples (if any)")
    parser.add_argument("--dataset_path", type=str, help="Dataset path for retrieving examples (if any)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write outputs")
    parser.add_argument("--server_url", type=str, default="http://localhost:6789", help="GRASP server URL")
    parser.add_argument("--kg", type=str, default="wikidata", help="Knowledge graph name (e.g., wikidata)")
    parser.add_argument("--timeout", type=int, default=180, help="Request timeout in seconds")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--continue", dest="continue_from", action="store_true", help="Continue from existing outputs")
    parser.add_argument("--dataset_name", type=str, default=None, help="Optional dataset name override for filenames")
    parser.add_argument("--oracle", action="store_true", help="Use oracle entity linking (if supported by the dataset)")
    parser.add_argument("--topic_entities_dataset", type=str, required=False, help="Dataset name (for DatasetFactory)")
    parser.add_argument("--examples", action="store_true", help="Use examples from examples_dataset to condition GRASP")
    parser.add_argument("--examples_dataset", type=str, help="Examples dataset name for retrieving examples")
    parser.add_argument("--examples_dataset_path", type=str, help="Examples dataset path for retrieving examples")
    args = parser.parse_args()

    create_console_logger()

    dataset = DatasetFactory.create_dataset(args.dataset, args.dataset_path)
    dataset_name = args.dataset_name or args.dataset
    base_path = os.path.join(args.output_dir, f"GRASP_{dataset_name}")

    if args.oracle and args.topic_entities_dataset:
        with open(args.topic_entities_dataset, "r") as f:
            topic_entities_dataset = json.load(f)
    elif args.oracle:
        print("Topic entities dataset must be provided when --oracle is set.")
        sys.exit(1)

    examples_db = None
    if args.examples and args.examples_dataset and args.examples_dataset_path:
        examples_dataset = DatasetFactory.create_dataset(args.examples_dataset, args.examples_dataset_path)
        from src.engine.qa.query_generator.query_db import QueryDb
        examples_db = QueryDb(examples_dataset) 
    elif args.examples:
        print("Examples dataset and path must be provided when --examples is set.")
        sys.exit(1)
    elif not args.examples and (args.examples_dataset or args.examples_dataset_path):
        print("Do not provide --examples_dataset or --examples_dataset_path without --examples.")
        sys.exit(1)
    
    if not args.overwrite and os.path.exists(base_path + "_results.json") and not args.continue_from:
        print(f"Outputs already exist at {base_path}_*.json; use --overwrite to regenerate or --continue to continue from where your left off.")
        sys.exit(0)

    if not args.continue_from:
        start_idx = 0
        results: List[Dict[str, Any]] = []
        results_full: List[Dict[str, Any]] = []
        eval_results: List[Dict[str, Any]] = []
        aggregate_metrics: Dict[str, float] = {}
        success_count = 0
        total_elapsed = 0.0
    else:
        if not os.path.exists(base_path + "_results.json") or not os.path.exists(base_path + "_results_full.json") or not os.path.exists(base_path + "_metrics.json"):
            print(f"Cannot continue; some output files are missing at {base_path}_*.json.")
            sys.exit(1)
        with open(base_path + "_results.json", "r") as f:
            results = json.load(f)
        with open(base_path + "_results_full.json", "r") as f:
            results_full = json.load(f)
        with open(base_path + "_metrics.json", "r") as f:
            aggregate_metrics = json.load(f)
        success_count = aggregate_metrics.get("QUESTIONS", 0)
        total_elapsed = aggregate_metrics.get("TIME", 0.0)
        start_idx = len(results)

    for idx in tqdm(range(start_idx, len(dataset))):
        entry = dataset[idx]
        question = dataset.get_question(entry)
        if args.oracle:
            entry = topic_entities_dataset[idx]
            topic_entities: Dict[str, str] = entry['topic_entities']
            topic_entities_str = ", ".join(
                [f"{ent} ({uri})" for ent, uri in topic_entities.items()]
            )
            topic_entities = topic_entities_str
        else:
            topic_entities = None

        if args.examples and examples_db is not None:
            example_questions, example_queries = examples_db.get_relevant_queries(question, top_k=3)
            examples  = "\n".join(
                [f"Question: {q}\nSPARQL:\n{sq}\n" for q, sq in zip(example_questions, example_queries)]
            )
        else:
            examples = None

        try:
            output = call_grasp_api(args.server_url, question, args.kg, args.timeout, topic_entities=topic_entities, examples=examples)

            sparql = extract_sparql(output)
            
            elapsed = output.get("elapsed") if isinstance(output, dict) else None

            metrics = extract_metrics(output)
            merge_metrics(aggregate_metrics, metrics)

            record = {"question": question, "sparql": sparql, "elapsed": elapsed, "metrics": metrics}
            record_full = {
                "question": question,
                "sparql": sparql,
                "output": output,
                "messages": output.get("messages") if isinstance(output, dict) else None,
                "elapsed": elapsed,
                "metrics": metrics,
                "error": output.get("error") if isinstance(output, dict) else None,
            }

            results.append(record)
            results_full.append(record_full)

            success_count += 1
            total_elapsed += float(elapsed)

        except Exception as exc:  # noqa: BLE001
            print("Error processing question:", question)
            err_msg = str(exc)
            record = {"question": question, "sparql": None, "error": err_msg}
            results.append(record)
            results_full.append(record)

        metrics_summary = {k: v for k, v in aggregate_metrics.items()}
        metrics_summary["QUESTIONS"] = success_count
        metrics_summary["TIME"] = total_elapsed
        metrics_summary["TIME_PER_QUESTION"] = total_elapsed / success_count if success_count else 0

        persist_all(base_path, results, results_full, metrics_summary)

    print(f"Saved outputs to {base_path}_*.json")


if __name__ == "__main__":
    main()
