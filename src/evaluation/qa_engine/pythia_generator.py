import os
import sys
import json
import argparse
import tempfile
import time
import requests

from tqdm import tqdm
from typing import Any, Dict, List

from src.engine.config import CONFIG
from src.datasets.dataset import DatasetFactory
from src.utils import execute_sparql_query, endpoints_fill_parse_args
from src.logging import create_logger, create_console_logger
from src.metrics import KgaqaTracker, get_kgaqa_tracker

from src.engine.agent.agents.agent_tools import AgentWithTools, AvailableTools


# -----------------------------
# ----- Metrics and files -----
# -----------------------------

def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def merge_metrics(total: Dict[str, float], metrics: Dict[str, Any]) -> None:
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            total[key] = total.get(key, 0.0) + float(val)

def persist_all(
    base_path: str,
    results: List[Dict[str, Any]],
    results_full: List[Dict[str, Any]],
    metrics_summary: Dict[str, Any],
) -> None:
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
    
    save_targets = [
        (results, base_path + "_results.json"),
        (results_full, base_path + "_results_full.json"),
        (metrics_summary, base_path + "_metrics.json"),
        # (eval_results, base_path + "_results_eval.json"),
        # (eval_metrics_summary, base_path + "_results_eval_metrics.json"),
    ]
    for data, path in save_targets:
        atomic_write(path, data)
        
# -------------------------------------
# ----- Setup auxiliary endpoints -----
# -------------------------------------
        
from src.utils import setup_graph_tool, setup_graphdb   

# ----------------
# ----- Main -----
# ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate results for Pythia.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--index_name", type=str, required=True, help="Knowledge graph index name (e.g., wikidata, dbpedia2016, etc.)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write outputs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--continue", dest="continue_from", action="store_true", help="Continue from existing outputs")
    parser.add_argument("--oracle", action="store_true", help="Use oracle entity linking (if supported by the dataset)")
    parser.add_argument("--bela", action="store_true", help="Use oracle entity linking (if supported by the dataset)")
    parser.add_argument("--topic_entities_dataset", type=str, help="Dataset")
    parser.add_argument("--no_find", action="store_true", help="Disable the use of the find tool in Pythia")
    parser.add_argument("--no_graph_search", action="store_true", help="Disable the use of the graph search tool in Pythia")
    parser.add_argument("--no_stepwise_search", action="store_true", help="Disable the use of the stepwise search tool in Pythia")
    parser.add_argument("--no_empty_query_investigation", action="store_true", help="Disable the investigation of empty query results in Pythia")
    parser.add_argument("--examples", action="store_true", help="Use examples from examples_dataset to condition GRASP")
    parser.add_argument("--examples_dataset", type=str, help="Examples dataset name for retrieving examples")
    parser.add_argument("--examples_dataset_path", type=str, help="Examples dataset path for retrieving examples")
    parser.add_argument("--end_idx", type=int, help="Index of the last question to process (exclusive)")
    parser.add_argument("--llm_provider", type=str, required=False, default="openai", help="The AI provider to use. Default is 'openai'.")
    parser.add_argument("--llm_model", type=str, required=False, default="gpt-4.1-mini", help="The AI provider to use. Default is 'gpt-4.1-mini'.")
    parser.add_argument("--agent_max_steps", type=int, required=False, default=10, help="Maximum number of steps for the agent to take when generating a SPARQL query. Default is 10.")
    parser.add_argument("--agent_grasp_prompt", action="store_true", help="Use the GRASP prompt.")
    parser.add_argument("--agent_basic_prompt", action="store_true", help="Use the basic prompt.")
    parser.add_argument("--agent_basic_relational_prompt", action="store_true", help="Use the basic relational prompt.")
    parser.add_argument("--agent_no_examples_prompt", action="store_true", help="Use the basic no examples prompt.")
    endpoints_fill_parse_args(parser)
    
    args = parser.parse_args()
    CONFIG(args)

    create_console_logger()
    
    # ---------------------------
    # ----- Parse Arguments -----
    # ---------------------------

    dataset = DatasetFactory.create_dataset(args.dataset, args.dataset_path)
    dataset_name = args.dataset
    base_path = os.path.join(args.output_dir, f"PYTHIA_{dataset_name}")
    
    kg = dataset.get_knowledge_graph()
    kg.load(os.path.join(CONFIG().get("index_dir"), args.index_name))
    kg_index_name = args.index_name
    
    if args.oracle and args.bela:
        print("Cannot use both --oracle and --bela at the same time.")
        sys.exit(1)

    if args.oracle and args.topic_entities_dataset:
        with open(args.topic_entities_dataset, "r") as f:
            topic_entities_dataset = json.load(f)
    elif args.oracle:
        print("Topic entities dataset must be provided when --oracle is set.")
        sys.exit(1)
    
    if args.bela and args.topic_entities_dataset:
        with open(args.topic_entities_dataset, "r") as f:
            topic_entities_dataset = json.load(f)
    elif args.bela:
        print("Topic entities dataset must be provided when --bela is set.")
        sys.exit(1)

    examples_dataset_name = None
    examples_dataset_path = None
    if args.examples and args.examples_dataset and args.examples_dataset_path:
        examples_dataset_name = args.examples_dataset
        examples_dataset_path = args.examples_dataset_path
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
        
    # --------------------------
    # ----- Prepare Pythia -----
    # --------------------------
    
    # print(f"Setting up GraphDB for graph: {kg_index_name}")
    # setup_graphdb(kg.value.endpoint)
    
    # print(f"Setting up graph tool for graph: {kg_index_name}")
    # setup_graph_tool(kg_index_name)
    
    if args.no_find:
        print("WARNING: Running with --no-find; the find tool will be disabled, which may lead to significantly worse performance.")
        agent_tools = [AvailableTools.STEPWISE_SEARCH, AvailableTools.GRAPH_SEARCH, AvailableTools.GET_PREDICATES]
    elif args.no_stepwise_search:
        print("WARNING: Running with --no-stepwise-search; the stepwise search tool will be disabled, which may lead to significantly worse performance.")
        agent_tools = [AvailableTools.FIND_ANCHORS, AvailableTools.GRAPH_SEARCH, AvailableTools.GET_PREDICATES]
    elif args.no_graph_search:
        print("WARNING: Running with --no-graph-search; the graph search tool will be disabled, which may lead to significantly worse performance.")
        agent_tools = [AvailableTools.FIND_ANCHORS, AvailableTools.STEPWISE_SEARCH, AvailableTools.GET_PREDICATES]
    else:
        agent_tools = [AvailableTools.FIND_ANCHORS, AvailableTools.STEPWISE_SEARCH, AvailableTools.GRAPH_SEARCH, AvailableTools.GET_PREDICATES]
    agent = AgentWithTools(kg, 
                           kg_index_name, 
                           tools=agent_tools,
                           db_dataset_name=examples_dataset_name,
                           db_dataset_path=examples_dataset_path,
                           enable_explanation_of_empty_results=not args.no_empty_query_investigation)
    
    if args.end_idx is not None:
        print(f"Limiting generation to the first {args.end_idx} questions (exclusive).")
        end_idx = args.end_idx
    else:
        end_idx = len(dataset)
    
    # --------------------
    # ----- Generate -----
    # --------------------

    for idx in tqdm(range(start_idx, end_idx)):
        entry = dataset[idx]
        question = dataset.get_question(entry)
        gold_entity_uris = None
        bela_entity_uris = None
        if args.oracle:
            oracle_entry = topic_entities_dataset[idx]
            topic_entities: Dict[str, str] = oracle_entry['topic_entities']
            topic_entities_str = ", ".join(
                [f"{ent} ({uri})" for uri, ent in topic_entities.items()]
            )
            # topic_entities = topic_entities_str
            gold_entity_uris = [uri for uri, ent in topic_entities.items()]
        elif args.bela:
            selected_bela_entry = None
            for bela_entry in topic_entities_dataset[1:]: # Skip header
                if question == bela_entry['question']:
                    selected_bela_entry = bela_entry
                    break
            if selected_bela_entry is None:
                print(f"Question mismatch between main dataset and BELA dataset at index {idx}.")
                sys.exit(1)
            bela_entry = selected_bela_entry
            topic_entities = None
            bela_entity_uris: List[str]= bela_entry['predictions']            

        try:
            output = agent.answer(
                question,
                gold_topic_entity_uris=gold_entity_uris,
                bela_topic_entity_uris=bela_entity_uris,
                stopwatch=True,
                sparql_only=False,
            )

            sparql = output['sparql']
            elapsed = output['elapsed']
            metrics = output['metrics']
            steps = output['steps']
            tool_calls_count = output['tool_calls_count']
            tool_calls_count_time = output['tool_calls_time']
            messages = output['messages']
            used_entities = output['used_entities']
            used_classes = output['used_classes']
            found_entities = output['found_entities']
            found_classes = output['found_classes']
            graph_search_start_end = output['graph_search_start_end']
            graph_search_tuples_returned = output['graph_search_tuples_returned']
            empty_due_to_filters_count = output['empty_due_to_filters_count']
            empty_due_to_invalid_triples_count = output['empty_due_to_invalid_triples_count']
            empty_due_to_invalid_combination_count = output['empty_due_to_invalid_combination_count']
            empty_due_to_select_vars_count = output['empty_due_to_select_vars_count']
            empty_unknown_count = output['empty_unknown_count']
            
            merge_metrics(aggregate_metrics, metrics)
            
            aggregate_metrics["steps"] = aggregate_metrics.get("steps", 0) + steps
            aggregate_metrics["find_calls"] = aggregate_metrics.get("find_calls", 0) + tool_calls_count['retrieve_entities_and_classes'] if 'retrieve_entities_and_classes' in tool_calls_count else 0
            aggregate_metrics["get_predicates_for_node"] = aggregate_metrics.get("get_predicates_for_node", 0) + tool_calls_count['get_predicates_for_node'] if 'get_predicates_for_node' in tool_calls_count else 0
            aggregate_metrics["stepwise_search_calls"] = aggregate_metrics.get("stepwise_search_calls", 0) + tool_calls_count['beam_search'] if 'beam_search' in tool_calls_count else 0
            aggregate_metrics["directed_exploration_calls"] = aggregate_metrics.get("directed_exploration_calls", 0) + tool_calls_count['bidirectional_bfs'] if 'bidirectional_bfs' in tool_calls_count else 0
            aggregate_metrics["execute_query_calls"] = aggregate_metrics.get("execute_query_calls", 0) + tool_calls_count['execute_query'] if 'execute_query' in tool_calls_count else 0
            aggregate_metrics["empty_due_to_filters_count"] = empty_due_to_filters_count
            aggregate_metrics["empty_due_to_invalid_triples_count"] = empty_due_to_invalid_triples_count
            aggregate_metrics["empty_due_to_invalid_combination_count"] = empty_due_to_invalid_combination_count
            aggregate_metrics["empty_due_to_select_vars_count"] = empty_due_to_select_vars_count
            aggregate_metrics["empty_unknown_count"] = empty_unknown_count
            record = {
                "question": question, 
                "sparql": sparql, 
                "elapsed": elapsed, 
                "metrics": metrics,
                "steps": steps,
                "tool_calls_count": tool_calls_count,
                "tool_calls_time": tool_calls_count_time,
                "used_entities": used_entities,
                "used_classes": used_classes,
            }
            record_full = {
                "question": question,
                "sparql": sparql,
                "messages": messages,
                "elapsed": elapsed,
                "metrics": metrics,
                "steps": steps,
                "tool_calls_count": tool_calls_count,
                "tool_calls_time": tool_calls_count_time,
                "used_entities": used_entities,
                "used_classes": used_classes,
                "found_entities": found_entities,
                "found_classes": found_classes,
                "graph_search_start_end": graph_search_start_end,
                "graph_search_tuples_returned": graph_search_tuples_returned,
            }

            results.append(record)
            results_full.append(record_full)

            success_count += 1
            total_elapsed += float(elapsed)

        except Exception as exc:  # noqa: BLE001
            raise exc
            # print("Error processing question:", question)
            # print(str(exc))
            # err_msg = str(exc)
            # record = {"question": question, "sparql": None, "error": err_msg}
            # results.append(record)
            # results_full.append(record)

        metrics_summary = {k: v for k, v in aggregate_metrics.items()}
        metrics_summary["QUESTIONS"] = success_count
        metrics_summary["TIME"] = total_elapsed
        metrics_summary["TIME_PER_QUESTION"] = total_elapsed / success_count if success_count else 0
        metrics_summary["LLM_INPUTS_PER_QUESTION"] = metrics_summary.get("LLM_INPUTS", 0) / success_count if success_count else 0
        metrics_summary["LLM_OUTPUTS_PER_QUESTION"] = metrics_summary.get("LLM_OUTPUTS", 0) / success_count if success_count else 0
        metrics_summary["AVG_STEPS_PER_QUESTION"] = metrics_summary.get("steps", 0) / success_count if success_count else 0
        metrics_summary["AVG_FIND_CALLS_PER_QUESTION"] = metrics_summary.get("find_calls", 0) / success_count if success_count else 0
        metrics_summary["AVG_GET_PREDICATES_FOR_NODE_CALLS_PER_QUESTION"] = metrics_summary.get("get_predicates_for_node", 0) / success_count if success_count else 0
        metrics_summary["AVG_STEPWISE_SEARCH_CALLS_PER_QUESTION"] = metrics_summary.get("stepwise_search_calls", 0) / success_count if success_count else 0
        metrics_summary["AVG_DIRECTED_EXPLORATION_CALLS_PER_QUESTION"] = metrics_summary.get("directed_exploration_calls", 0) / success_count if success_count else 0
        metrics_summary["AVG_EXECUTE_QUERY_CALLS_PER_QUESTION"] = metrics_summary.get("execute_query_calls", 0) / success_count if success_count else 0
        metrics_summary["EMPTY_DUE_TO_FILTERS_COUNT"] = metrics_summary.get("empty_due_to_filters_count", 0)
        metrics_summary["EMPTY_DUE_TO_INVALID_TRIPLES_COUNT"] = metrics_summary.get("empty_due_to_invalid_triples_count", 0)
        metrics_summary["EMPTY_DUE_TO_INVALID_COMBINATION_COUNT"] = metrics_summary.get("empty_due_to_invalid_combination_count", 0)
        metrics_summary["EMPTY_DUE_TO_SELECT_VARS_COUNT"] = metrics_summary.get("empty_due_to_select_vars_count", 0)
        metrics_summary["EMPTY_UNKNOWN_COUNT"] = metrics_summary.get("empty_unknown_count", 0)

        persist_all(base_path, results, results_full, metrics_summary)

    print(f"Saved outputs to {base_path}_*.json")


if __name__ == "__main__":
    main()
