import os
import argparse
import requests
import tqdm
import time
import json
import itertools

from typing import List, Dict, Tuple, Set

from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from src.engine.config import CONFIG
from src.elelem.provider import Provider, ProviderFactory
from src.engine.qa.kg_explorer.stepwise_search_manager import StepwiseSearchManager, StepwiseSearchMetrics, StepwiseSearchResultEnum
from src.engine.qa.kg_explorer.kg_path import KgPath
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.knowledge_graphs.knowledge_graph import KnowledgeGraph
from src.logging import log, LogComponent, LogLevel , create_console_logger
from src.utils import embed, endpoints_fill_parse_args, setup_graphdb


def load_dataset(dataset_path: str):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    data = []
    for sample in dataset:
        question = sample.get("question", "")
        named_entities = sample.get("named_entities", [])
        classes = sample.get("classes", [])
        
        if len(named_entities) + len(classes) != 1:
            continue  # We only handle questions with a single topic entity. This makes sure that known-uknown paths are useful.
        
        paths = sample.get("known_to_unknown_paths_triples", [])
            
        for path in paths:
            start_uri = path[0][0]
            if not start_uri.startswith("http"):
                raise ValueError(f"Invalid start URI: {start_uri}")
            
            triples_as_tuples = convert_triples_to_tuples(path)
            
            data.append({
                "question": question,
                "start_uri": start_uri,
                "tuples": triples_as_tuples,
            })
    return data


def convert_triples_to_tuples(triples_path: List[List[str]]) -> List[Tuple[str, str, str]]:
    path_tuples = []
    for triple in triples_path:
        # print(triple)
        if triple[1][0] == '~':
            path_tuples.append((triple[2], triple[1][1:], triple[0]))
        else:
            path_tuples.append((triple[0], triple[1], triple[2]))
    return path_tuples


def get_variables(tuples: List[Tuple[str, str, str]]) -> Set[str]:
    """Extract all unique variable names (strings starting with '?') from the tuples."""
    variables = set()
    for t in tuples:
        for element in t:
            if element.startswith("?"):
                variables.add(element)
    return variables


def apply_mapping(tuples: List[Tuple[str, str, str]], mapping: Dict[str, str]) -> Set[Tuple[str, str, str]]:
    """Apply a variable mapping to a list of tuples and return a set of transformed tuples."""
    mapped_tuples = set()
    for t in tuples:
        new_t = []
        for element in t:
            # If element is a variable, swap it using the mapping; otherwise keep it
            new_t.append(mapping.get(element, element))
        mapped_tuples.add(tuple(new_t))
    return mapped_tuples

class Metrics:
    def __init__(self, name):
        self.name = name
        self.tested_paths = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.macro_f1 = 0.0
        self.macro_recall = 0.0
        self.macro_precision = 0.0
        self.time = 0.0
        self.steps = 0
        self.sparql_time = 0.0
        self.sparql_calls = 0
        self.llm_time = 0.0
        self.llm_calls = 0
        self.llm_inputs = 0
        self.llm_outputs = 0
        self.failed: List[str] = []
        self.found = 0
        self.cancel = 0
        self.max_steps_reached = 0
        self.dead_end_reached = 0
        
        # results
        self.per_entry_results = []

    def calculate_metrics(self) -> Tuple[float, float, float, float]:
        found = self.tp / self.tested_paths if self.tested_paths > 0 else 0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return found, precision, recall, f1
    
    def dict(self) -> Dict:
        found, precision, recall, f1 = self.calculate_metrics()
        if self.llm_calls > 0:
            return {
                "tested_paths": self.tested_paths,
                "found": found,
                "em_f1": f1,
                "em_recall": recall,
                "em_precision": precision,
                "macro_f1": self.macro_f1 / self.tested_paths if self.tested_paths > 0 else 0,
                "macro_recall": self.macro_recall / self.tested_paths if self.tested_paths > 0 else 0,
                "macro_precision": self.macro_precision / self.tested_paths if self.tested_paths > 0 else 0,
                "returned_paths": (self.tp + self.fp),
                "found-enum": self.found,
                "cancel": self.cancel,
                "max_steps_reached": self.max_steps_reached,
                "dead_end_reached": self.dead_end_reached,
                "predictions_per_query": (self.tp + self.fp) / self.tested_paths if self.tested_paths > 0 else 0,
                "time": self.time,
                "average_steps": self.steps / self.tested_paths if self.tested_paths > 0 else 0,
                "sparql_time": self.sparql_time,
                "sparql_calls": self.sparql_calls,
                "llm_time": self.llm_time,
                "llm_calls": self.llm_calls,
                "time_per_path": self.time / self.tested_paths if self.tested_paths > 0 else 0,
                "sparql_time_per_path": self.sparql_time / self.tested_paths if self.tested_paths > 0 else 0,
                "llm_time_per_path": self.llm_time / self.tested_paths if self.tested_paths > 0 else 0,
                "llm_inputs": self.llm_inputs,
                "llm_outputs": self.llm_outputs,
                "llm_input_per_path": self.llm_inputs / self.tested_paths if self.tested_paths > 0 else 0,
                "llm_output_per_path": self.llm_outputs / self.tested_paths if self.tested_paths > 0 else 0,
                "tp": self.tp,
                "fp": self.fp,
                "fn": self.fn,
                "failed_paths": self.failed,
            }
        else:
            return {
                "tested_paths": self.tested_paths,
                "found": found,
                "em_f1": f1,
                "em_recall": recall,
                "em_precision": precision,
                "macro_f1": self.macro_f1 / self.tested_paths if self.tested_paths > 0 else 0,
                "macro_recall": self.macro_recall / self.tested_paths if self.tested_paths > 0 else 0,
                "macro_precision": self.macro_precision / self.tested_paths if self.tested_paths > 0 else 0,
                "returned_paths": (self.tp + self.fp),
                "found-enum": self.found,
                "cancel": self.cancel,
                "max_steps_reached": self.max_steps_reached,
                "dead_end_reached": self.dead_end_reached,
                "predictions_per_query": (self.tp + self.fp) / self.tested_paths if self.tested_paths > 0 else 0,
                "time": self.time,
                "time_per_path": self.time / self.tested_paths if self.tested_paths > 0 else 0,
                "tp": self.tp,
                "fp": self.fp,
                "fn": self.fn,
                "failed_paths": self.failed,
            }


def save_scores(metrics: Metrics, file_path):
    with open(file_path, "w") as file:
        json.dump(metrics.dict(), file, indent=4)
        

def save_results(results: List[Dict], file_path):
    with open(file_path, "w") as file:
        json.dump(results, file, indent=4)
        
        
def evaluate_and_save_results(question: str, result: StepwiseSearchResultEnum, paths: List[KgPath], path_metrics: StepwiseSearchMetrics, metrics: Metrics, metrics_output_file, results_output_file):    
    metrics.time += path_metrics.time
    metrics.tested_paths += 1
    metrics.sparql_time += path_metrics.sparql_time
    metrics.sparql_calls += path_metrics.sparql_calls
    metrics.llm_time += path_metrics.llm_time
    metrics.llm_calls += path_metrics.llm_calls
    metrics.llm_inputs += path_metrics.llm_inputs
    metrics.llm_outputs += path_metrics.llm_outputs
    metrics.steps += path_metrics.steps
    paths_values_list = [kg_enum.value.get_values_for_triples(path.get_triples_string(readable=False), k=1000) for path in paths]
    match_found = False
    if result == StepwiseSearchResultEnum.FOUND:
        metrics.found += 1
    if result == StepwiseSearchResultEnum.CANCEL:
        metrics.cancel += 1
    if result == StepwiseSearchResultEnum.MAX_STEPS_REACHED:
        metrics.max_steps_reached += 1
    if result == StepwiseSearchResultEnum.DEAD_END_REACHED:
        metrics.dead_end_reached += 1
    for i, path_values in enumerate(paths_values_list):
        column_names = list(path_values[0].keys()) if len(path_values) > 0 else []
        column_values: List[List[str]] = []
        for col_name in column_names:
            column_values.append([val[col_name] for val in path_values])
        for column in column_values:
            tp = len(set(column) & set(expected_values_last_column))
            fp = len(set(column) - set(expected_values_last_column))
            fn = len(set(expected_values_last_column) - set(column))
            if tp > 0 and fp == 0 and fn == 0:
                match_found = True
                break
        if match_found:
            break
    
    max_recall = 0.0
    max_precision = 0.0
    max_f1 = 0.0
    for i, path_values in enumerate(paths_values_list):
        column_names = list(path_values[0].keys()) if len(path_values) > 0 else []
        column_values: List[List[str]] = []
        for col_name in column_names:
            column_values.append([val[col_name] for val in path_values])
        for column in column_values:
            tp = len(set(column) & set(expected_values_last_column))
            fp = len(set(column) - set(expected_values_last_column))
            fn = len(set(expected_values_last_column) - set(column))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > max_f1:
                max_f1 = f1
                max_recall = recall
                max_precision = precision

    # # print(f"  Shortest paths found: {shortest_paths_tuples_list}")
    if match_found:
        metrics.tp += 1
        print(f"\t{metrics.name}: ✅ in {path_metrics.time:.4f} seconds")
    else:
        print(f"\t{metrics.name}: ❌ in {path_metrics.time:.4f} seconds")
        metrics.fn += 1
        metrics.failed.append(start_uri)
    metrics.fp += len(paths) - (1 if match_found else 0)
    
    metrics.macro_f1 += max_f1
    metrics.macro_precision += max_precision
    metrics.macro_recall += max_recall
    
    save_scores(metrics, metrics_output_file)
    
    per_entry_results = {
        "question": question,
        "found": match_found,
        "macro_f1": max_f1,
        "correct_path": expected_tuples,
        "selected_paths": [path.get_tuples() for path in paths],
        "correct_triples_string": expected_tuples_as_triples,
        "selected_triples_strings": [path.get_triples_string(readable=False) for path in paths],
    }
    metrics.per_entry_results.append(per_entry_results)
    save_results(metrics.per_entry_results, results_output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate results for Pythia.")
    parser.add_argument("--steps", type=int, required=True, help="Index of the last question to process (exclusive)")
    parser.add_argument("--window_size", type=int, required=True, help="Window size for the stepwise search")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing results or skip already processed datasets")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the evaluation results")
    endpoints_fill_parse_args(parser)
    
    args = parser.parse_args()
    CONFIG(args)
    
    create_console_logger()
    
    MAX_STEPS = args.steps
    WINDOW_SIZE = args.window_size
    ENABLE_BACKTRACKING = False
    ENABLE_LIMIT = True
    ENABLE_PARTIAL_RESULTS = False
    OVERWRITE = False
    
    datasets = [
        ('../../../datasets/_graph_search_evaluation/QALD-9_graph_exploration_evaluation.jsonl',     KnowledgeGraphs.DBPEDIA10,  "dbpedia10"),
        ('../../../datasets/_graph_search_evaluation/QALD-10_graph_exploration_evaluation.jsonl',    KnowledgeGraphs.WIKIDATA,   "wikidata"),
        ('../../../datasets/_graph_search_evaluation/WebQSP_graph_exploration_evaluation.jsonl',     KnowledgeGraphs.FREEBASE,   "freebase"),
    ]
    
    for dataset_path, kg_enum, graph_string in datasets:
        print(f"Setting up GraphDB for graph: {graph_string}")
        setup_graphdb(kg_enum.value.endpoint)
        
        print(f"Setting up knowledge graph: {kg_enum.name}")
        kg_enum.load(os.path.join(CONFIG().get('index_dir'), graph_string))
        
        print(f"Loading dataset from: {dataset_path}")
        data = load_dataset(dataset_path)
        
        print(f"Initializing StepwiseSearchManager for graph: {graph_string}")
        stepwise_search_manager = StepwiseSearchManager(kg_enum, ProviderFactory.create_from_config(CONFIG()))
        
        total_paths = len(data)
        tested_paths = 0
        
        invalid_connections = 0
        
        metrics_stepwise_search = Metrics("stepwise_search")
        file_suffix = dataset_path.split("/")[-1].replace("graph_exploration_evaluation.jsonl", f"stepwise_search_evaluation_gpt41_{MAX_STEPS}_{WINDOW_SIZE}_{ENABLE_BACKTRACKING}_{ENABLE_LIMIT}_{ENABLE_PARTIAL_RESULTS}")
        stepwise_search_metrics_output_file = os.path.join(args.output_dir, file_suffix + "_metrics.json")
        stepwise_search_results_output_file = os.path.join(args.output_dir, file_suffix + "_results.json")
        
        if not OVERWRITE:
            if os.path.exists(stepwise_search_metrics_output_file) or os.path.exists(stepwise_search_results_output_file):
                print(f"Stepwise search evaluation already exists for dataset {dataset_path}. Skipping...")
                continue
        
        print(f"Evaluating {total_paths} paths...")
        for idx, item in tqdm.tqdm(enumerate(data), desc="Evaluating paths", total=total_paths):
            question = item["question"]
            start_uri = item["start_uri"]
            expected_tuples = item["tuples"]
            
            print(f"Expecting path: {expected_tuples}")
            
            expected_tuples_as_triples = ""
            for t in expected_tuples:
                expected_tuples_as_triples += f"<{t[0]}> " if not t[0].startswith("?") else f"{t[0]} "
                expected_tuples_as_triples += f"<{t[1]}> " if not t[1].startswith("?") else f"{t[1]} "
                expected_tuples_as_triples += f"<{t[2]}> .\n" if not t[2].startswith("?") else f"{t[2]} .\n"
            valid = kg_enum.value.are_triples_valid(expected_tuples_as_triples)
            if not valid:
                print(f"Warning: Expected tuples contain invalid triples for KG {kg_enum.name}: {expected_tuples_as_triples}")
                invalid_connections += 1
                continue
            
            expected_values = kg_enum.value.get_values_for_triples(expected_tuples_as_triples, k=1000)
            last_var_of_expected_values = list(expected_values[0].keys())[-1]
            expected_values_last_column = [val[last_var_of_expected_values] for val in expected_values]
            
            # exit(1)
            tested_paths += 1
            
            result, paths, metrics = stepwise_search_manager.search(question, start_uri, max_steps=MAX_STEPS, enable_backtracking=ENABLE_BACKTRACKING, enable_limit=ENABLE_LIMIT, return_partial_results=ENABLE_PARTIAL_RESULTS, window_size=WINDOW_SIZE)
            # for path in paths:
            #     values = kg_enum.value.get_values_for_triples(path.get_triples_string(readable=False), k=1000)
            evaluate_and_save_results(question, result, paths, metrics, metrics_stepwise_search, stepwise_search_metrics_output_file, stepwise_search_results_output_file)
            
            # exit(1)
        
        complete_results = {
            "total_paths": total_paths,
            "tested_paths": tested_paths,
            "invalid_connections": invalid_connections,
            "stepwise_search": metrics_stepwise_search.dict(),
        }
        
        with open(dataset_path.replace("graph_exploration_evaluation.jsonl", f"stepwise_search_complete_results_{MAX_STEPS}_{ENABLE_BACKTRACKING}_{ENABLE_LIMIT}_{ENABLE_PARTIAL_RESULTS}.json"), "w") as file:
            json.dump(complete_results, file, indent=4)