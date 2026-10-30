import os
import sys
import json
import tempfile

from tqdm import tqdm
from typing import Any, Dict, List

from src.config import LLM_SETTINGS, PythiaArgParser, get_configurations_dump
from src.logging import create_console_logger
from src.dataloaders.dataset import DatasetFactory
from src.core.agent.pythia_agent import PythiaAgent

# -----------------------------
# ----- Metrics and files -----
# -----------------------------

def merge_metrics(total: Dict[str, float], metrics: Dict[str, Any]) -> None:
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            total[key] = total.get(key, 0.0) + float(val)

def persist_all(
    args,
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
        
    config_data = get_configurations_dump()
    config_data['args'] = vars(args)
    
    save_targets = [
        (results, get_results_filepath(args)),
        (results_full, get_results_full_filepath(args)),
        (metrics_summary, get_metrics_filepath(args)),
        (config_data, get_config_filepath(args)),
    ]
    for data, path in save_targets:
        atomic_write(path, data)
        
def construct_base_output_path(args) -> str:
    end_idx_str = f"_end{args.end_idx}" if args.end_idx is not None else ""
    linking_str = "_lo" if args.oracle else "_lb" if args.bela else "_ln"
    if AGENT is None:
        raise ValueError("AGENT is not initialized. Please ensure that AGENT is set before calling this function.")
    tools_str = AGENT.get_tools_str()
    prompt_str = AGENT.get_prompt_str()
    base_output_path = os.path.join(args.output_dir, LLM_SETTINGS.model, f"p-{prompt_str}_t-{tools_str}", f"PYTHIA{linking_str}{end_idx_str}")
    return base_output_path

def get_metrics_filepath(args) -> str:
    return construct_base_output_path(args) + "_metrics.json"

def get_results_filepath(args) -> str:
    return construct_base_output_path(args) + "_results.json"

def get_results_full_filepath(args) -> str:
    return construct_base_output_path(args) + "_results_full.json"

def get_config_filepath(args) -> str:
    return construct_base_output_path(args) + "_config.json"

# ----------------
# ----- Main -----
# ----------------

def main() -> None:
    parser = PythiaArgParser(description="Answer questions with PYTHIA over a given dataset.\n\nRequires at least the following arguments:\
                             \n --dataset,\n --dataset_path,\n --graph,\n [--rdf_endpoint_url] (if you have not configured a default endpoint for the graph)\
                             \n\nIt also requires setting the following environment variables:\
                             \n - API key for the LLM provider of your choice (default is: OPENAI_API_KEY).\n - GRAPH_SEARCH_SERVER to point to the graph search server.")
    
    # -----------------------------
    # ----- BENCHMARK OPTIONS -----
    # -----------------------------
    # Dataset (Input) & Output
    DatasetFactory.fill_parse_args(parser)
    generator_parser = parser.add_argument_group("Generation Settings")
    generator_parser.add_argument("--output_dir", type=str, required=False, default='./output', help="Directory to write outputs. Default: ./output")
    # Run Options
    generator_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs. Default: False")
    generator_parser.add_argument("--continue", dest="continue_from", action="store_true", help="Continue from existing outputs. Default: False")
    generator_parser.add_argument("--end_idx", type=int, help="Index of the last question to process (exclusive). If not provided, all questions will be processed.")
    # Topic Entities
    generator_parser.add_argument("--oracle", action="store_true", help="Use oracle entity linking (if supported by the dataset).")
    generator_parser.add_argument("--bela", action="store_true", help="Use oracle entity linking (if supported by the dataset).")
    generator_parser.add_argument("--topic_entities_dataset", type=str, help="Dataset containing topic entities. Required if --oracle or --bela is set. Should be a JSON file with the same number of entries as the main dataset, where each entry contains a 'topic_entities' field mapping entity URIs to their labels.")
    
    # --------------------------
    # ----- PYTHIA OPTIONS -----
    # --------------------------
    PythiaAgent.fill_parse_args(parser, graph_required=True)
    
    args = parser.parse_args()

    create_console_logger()
    
    # ---------------------------
    # ----- Parse Arguments -----
    # ---------------------------

    dataset = DatasetFactory.create_dataset(args.dataset, args.dataset_path)
    
    kg = dataset.get_knowledge_graph()
    kg.load()
    
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
        
    # --------------------------
    # ----- Prepare Pythia -----
    # --------------------------
    
    examples_dataset_name = None
    examples_dataset_path = None
    if args.examples_dataset and args.examples_dataset_path:
        examples_dataset_name = args.examples_dataset
        examples_dataset_path = args.examples_dataset_path
    elif args.examples_dataset or args.examples_dataset_path:
        print("You must provide both --examples_dataset and --examples_dataset_path if you want to use examples.")
        sys.exit(1)
    
    global AGENT
    AGENT = PythiaAgent(kg,
                        db_dataset_name=examples_dataset_name,
                        db_dataset_path=examples_dataset_path)
    
    # --------------------------------------------------
    # ----- Check files & overwrite/continue logic -----
    # --------------------------------------------------
    
    if not args.overwrite and os.path.exists(get_results_filepath(args)) and not args.continue_from:
        print(f"Outputs already exist at {get_results_filepath(args)}; use --overwrite to regenerate or --continue to continue from where your left off.")
        sys.exit(0)

    if not args.continue_from:
        start_idx = 0
        results: List[Dict[str, Any]] = []
        results_full: List[Dict[str, Any]] = []
        aggregate_metrics: Dict[str, float] = {}
        success_count = 0
        total_elapsed = 0.0
    else:
        if not os.path.exists(get_results_filepath(args)) or not os.path.exists(get_results_full_filepath(args)) or not os.path.exists(get_metrics_filepath(args)):
            print(f"Cannot continue; some output files are missing at {get_results_filepath(args)}.")
            sys.exit(1)
        with open(get_results_filepath(args), "r") as f:
            results = json.load(f)
        with open(get_results_full_filepath(args), "r") as f:
            results_full = json.load(f)
        with open(get_metrics_filepath(args), "r") as f:
            aggregate_metrics = json.load(f)
        success_count = aggregate_metrics.get("QUESTIONS", 0)
        total_elapsed = aggregate_metrics.get("TIME", 0.0)
        start_idx = len(results)
    
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
            output = AGENT.answer(
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

        persist_all(args, results, results_full, metrics_summary)

    print(f"Saved outputs to {construct_base_output_path(args)}_*.json")


if __name__ == "__main__":
    AGENT = None
    main()
