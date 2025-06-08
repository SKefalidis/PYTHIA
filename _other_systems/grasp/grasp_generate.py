import os
import sys
import json
import argparse
import tempfile
from typing import Any, Dict, List, Tuple

import requests
from tqdm import tqdm

def atomic_write(filepath: str, data: Any, mode: str = "w") -> None:
    dir_name = os.path.dirname(filepath)
    os.makedirs(dir_name, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode=mode, dir=dir_name, delete=False) as tmp_file:
        json.dump(data, tmp_file, indent=4)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_name = tmp_file.name
    os.replace(temp_name, filepath)

def save_to_file(run_results: Any, output_file_path: str) -> None:
    try:
        atomic_write(output_file_path, run_results)
    except Exception as e:
        print(f"Error saving files: {str(e)}")

def call_grasp_api(
    server_url: str,
    question: str,
    kg: str,
    task: str = "sparql-qa",
    timeout: int = 120,
) -> Dict[str, Any]:
    payload = {"task": task, "input": question, "knowledge_graphs": [kg]}
    response = requests.post(
        f"{server_url}/run",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def extract_sparql(resp: Dict[str, Any]) -> str | None:
    if "sparql" in resp and isinstance(resp.get("sparql"), str):
        return resp["sparql"]
    output = resp.get("output", {}) if isinstance(resp, dict) else {}
    return output.get("sparql") if isinstance(output, dict) else None


def extract_metrics(resp: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(resp, dict):
        return {}
    metrics = resp.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    output = resp.get("output")
    if isinstance(output, dict) and isinstance(output.get("metrics"), dict):
        return output["metrics"]
    return {}


def merge_metrics(total: Dict[str, float], metrics: Dict[str, Any]) -> None:
    for key, val in metrics.items():
        if not isinstance(val, (int, float)):
            continue
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

        gold_query = item.get("gold_query")
        if gold_query:
            summary["total_valid"] += 1
        else:
            summary["empty_gold_queries"] += 1

        for key, target in [
            ("tp", "total_tp"),
            ("fp", "total_fp"),
            ("fn", "total_fn"),
        ]:
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

    summary["f1"] = safe_ratio(2 * summary["total_tp"], 2 * summary["total_tp"] + summary["total_fp"] + summary["total_fn"])
    summary["precision"] = safe_ratio(summary["total_tp"], summary["total_tp"] + summary["total_fp"])
    summary["recall"] = safe_ratio(summary["total_tp"], summary["total_tp"] + summary["total_fn"])

    total_questions = max(summary.get("total", 0), 1)
    summary["average_time_per_question"] = safe_ratio(summary["total_time"], total_questions)
    summary["average_sparql_calls_per_question"] = safe_ratio(summary["total_sparql_calls"], total_questions)
    summary["average_sparql_time_per_question"] = safe_ratio(summary["total_sparql_time"], total_questions)
    summary["average_llm_calls_per_question"] = safe_ratio(summary["total_llm_calls"], total_questions)
    summary["average_llm_time_per_question"] = safe_ratio(summary["total_llm_time"], total_questions)
    summary["average_llm_inputs_per_question"] = safe_ratio(summary["total_llm_inputs"], total_questions)
    summary["average_llm_outputs_per_question"] = safe_ratio(summary["total_llm_outputs"], total_questions)

    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRASP text-to-SPARQL engine on a dataset.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the input dataset (JSON)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--server_url", type=str, default="http://localhost:6789", help="GRASP server URL")
    parser.add_argument("--kg", type=str, default="wikidata", help="Knowledge graph name (e.g., wikidata)")
    parser.add_argument("--task", type=str, default="sparql-qa", help="GRASP task (default: sparql-qa)")
    parser.add_argument("--timeout", type=int, default=180, help="Request timeout in seconds")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--dataset_name", type=str, default=None, help="Optional dataset name override for output filenames")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    args = parser.parse_args()

    with open(args.dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results: List[Dict[str, Any]] = []
    results_full: List[Dict[str, Any]] = []
    eval_results: List[Dict[str, Any]] = []
    aggregate_metrics: Dict[str, float] = {}

    dataset_name = args.dataset_name or os.path.splitext(os.path.basename(args.dataset_file))[0]
    base_path = os.path.join(args.output_dir, f"GRASP_{dataset_name}")

    if not args.overwrite and os.path.exists(base_path + "_results.json"):
        print(f"Output already exists at {base_path}_*.json; use --overwrite to regenerate")
        sys.exit(0)

    for idx, entry in enumerate(tqdm(dataset)):
        question = None
        if isinstance(entry, dict):
            question = entry.get("question") or entry.get("text") or entry.get("q")
        elif isinstance(entry, str):
            question = entry

        if not question:
            print(f"No question found in entry {idx}")
            continue

        gold_query = entry.get("query") if isinstance(entry, dict) else None

        try:
            output = call_grasp_api(args.server_url, question, args.kg, args.task, args.timeout)
            sparql = extract_sparql(output)
            elapsed = output.get("elapsed") if isinstance(output, dict) else None
            metrics = extract_metrics(output)
            merge_metrics(aggregate_metrics, metrics)

            record = {
                "question": question,
                "sparql": sparql,
                "elapsed": elapsed,
                "metrics": metrics,
            }

            record_full = {
                "question": question,
                "sparql": sparql,
                "output": output,
                "messages": output.get("messages") if isinstance(output, dict) else None,
                "elapsed": elapsed,
                "metrics": metrics,
                "error": output.get("error") if isinstance(output, dict) else None,
            }

            # lightweight evaluation: exact string match if gold query present
            exact_match = bool(gold_query and sparql and sparql.strip() == gold_query.strip())
            eval_entry = {
                "question": question,
                "generated_query": sparql,
                "gold_query": gold_query,
                "results": {
                    "tp": None,
                    "fp": None,
                    "fn": None,
                    "hits_at_1": int(exact_match) if gold_query else None,
                    "exact_match": exact_match if gold_query else False,
                },
                "metrics": metrics,
                "elapsed": elapsed,
            }

            results.append(record)
            results_full.append(record_full)
            eval_results.append(eval_entry)

            if args.debug:
                print(f"[{idx}] {question}\nSPARQL: {sparql}\n")

        except Exception as e:
            err_msg = str(e)
            print(f"Error for question {idx}: {err_msg}")
            record = {"question": question, "sparql": None, "error": err_msg}
            results.append(record)
            results_full.append(record)
            eval_results.append(
                {
                    "question": question,
                    "generated_query": None,
                    "gold_query": gold_query,
                    "results": {"tp": None, "fp": None, "fn": None, "hits_at_1": None, "exact_match": False},
                    "metrics": {},
                    "elapsed": None,
                    "error": err_msg,
                }
            )

    # aggregate metrics summary
    metrics_summary = {k: v for k, v in aggregate_metrics.items()}
    if "QUESTIONS" not in metrics_summary:
        metrics_summary["QUESTIONS"] = len(results)
    if "TIME" not in metrics_summary:
        total_time = sum(r.get("elapsed") for r in results if isinstance(r.get("elapsed"), (int, float)))
        metrics_summary["TIME"] = total_time
    metrics_summary["TIME_PER_QUESTION"] = (
        metrics_summary["TIME"] / metrics_summary["QUESTIONS"] if metrics_summary.get("QUESTIONS") else 0
    )

    eval_metrics_summary = compute_eval_summary(eval_results)

    save_to_file(results, base_path + "_results.json")
    save_to_file(results_full, base_path + "_results_full.json")
    save_to_file(metrics_summary, base_path + "_metrics.json")
    save_to_file(eval_results, base_path + "_results_eval.json")
    save_to_file(eval_metrics_summary, base_path + "_results_eval_metrics.json")

if __name__ == "__main__":
    main()
