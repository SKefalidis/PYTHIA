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
from src.engine.qa.kg_explorer.graph_search_manager import GraphSearchManager, GraphSearchMetrics
from src.engine.qa.kg_explorer.kg_path import KgPath
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.knowledge_graphs.knowledge_graph import KnowledgeGraph
from src.logging import log, LogComponent, LogLevel , create_console_logger
from src.utils import embed, setup_graph_tool, setup_graphdb, get_relative_path


def load_dataset(dataset_path: str):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    data = []
    for sample in dataset:
        question = sample.get("question", "")
        named_entities = sample.get("named_entities", [])
        classes = sample.get("classes", [])
        paths = sample.get("known_to_known_paths_triples", [])
        
        path_type = "unknown"
        for path in paths:
            start_uri = path[0][0]
            end_uri = path[-1][-1]
            
            if start_uri in named_entities and end_uri in named_entities:
                path_type = "named_to_named"
            elif start_uri in named_entities and end_uri in classes:
                path_type = "named_to_class"
            elif start_uri in classes and end_uri in named_entities:
                path_type = "class_to_named"
            elif start_uri in classes and end_uri in classes:
                path_type = "class_to_class"
                # print(f"Class to class path found: {path}")
            else:
                path_type = "unknown"
                # print(f"Unknown path type for start: {start_uri}, end: {end_uri}")
            
            triples_as_tuples = convert_triples_to_tuples(path)
            
            data.append({
                "question": question,
                "start_uri": start_uri,
                "end_uri": end_uri,
                "path_type": path_type,
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


def check_tuples_equality(tuples1: List[Tuple[str, str, str]], tuples2: List[Tuple[str, str, str]]) -> bool:
    """
    Check if two lists of triples are isomorphic.
    1. They must be the same length.
    2. They must have the same number of unique variables.
    3. There must exist a mapping of vars1 -> vars2 such that set(tuples1) == set(tuples2).
    """
    # 1. Quick check on length
    if len(tuples1) != len(tuples2):
        return False

    # 2. Extract variables
    vars1 = sorted(list(get_variables(tuples1)))
    vars2 = sorted(list(get_variables(tuples2)))

    # If number of unique variables differs, they cannot be structurally equal
    if len(vars1) != len(vars2):
        return False

    # 3. Brute-force check all permutations (Isomorphism check)
    # We try to map vars1 to every possible permutation of vars2
    target_set = set(tuples2)
    
    # If there are no variables, just compare the sets of constants
    if not vars1:
        return set(tuples1) == target_set

    for perm in itertools.permutations(vars2):
        # Create a mapping dictionary: {?x: ?y, ?z: ?a, ...}
        mapping = dict(zip(vars1, perm))
        
        # print(mapping)
        
        # Apply this mapping to the first list
        mapped_set = apply_mapping(tuples1, mapping)
        
        # Check if the transformed set matches the target set
        if mapped_set == target_set:
            return True

    return False


def rank_paths_by_embedding_similarity(question, kg, found_paths: List[KgPath]) -> List[KgPath]:
    if len(found_paths) == 0:
        return []
    
    question_embedding = embed(question, is_query=False)
    
    path_verbalizations = [path.verbalize_path(kg) for path in found_paths]
    path_embeddings = embed(path_verbalizations, is_query=False)
    
    path_similarities = cosine_similarity([question_embedding], path_embeddings)[0]
    ranked_paths = [path for _, path in sorted(zip(path_similarities, found_paths), key=lambda x: x[0], reverse=True)]
    
    return ranked_paths


cross_encoder_model = CrossEncoder('cross-encoder/stsb-roberta-large')
def rank_paths_with_cross_encoder(question, kg, found_paths: List[KgPath]) -> List[KgPath]:
    path_verbalizations = [path.verbalize_path(kg) for path in found_paths]
    
    # Prepare input for cross-encoder
    cross_encoder_inputs = [(question, path_verbalization) for path_verbalization in path_verbalizations]
    
    # Get similarity scores from cross-encoder
    global cross_encoder_model
    path_similarities = cross_encoder_model.predict(cross_encoder_inputs)
    
    ranked_paths = [path for _, path in sorted(zip(path_similarities, found_paths), key=lambda x: x[0], reverse=True)]
    
    return ranked_paths


def rank_paths_by_specificity(kg: KnowledgeGraph, found_paths: List[KgPath]) -> List[KgPath]:
    path_scores = []
    for path in found_paths:
        score = 0.0
        for part in path.parts:
            predicate = part.predicate
            predicate_popularity = kg.get_kg_component(predicate).incoming_edges_count + kg.get_kg_component(predicate).outgoing_edges_count
            score -= predicate_popularity
        score += path.popularity * 10.0
        path_scores.append(score)
    
    ranked_paths = [path for _, path in sorted(zip(path_scores, found_paths), key=lambda x: x[0], reverse=True)]
    
    return ranked_paths


class Metrics:
    def __init__(self, name):
        self.name = name
        self.tested_paths = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.time = 0.0
        self.sparql_time = 0.0
        self.sparql_calls = 0
        self.llm_time = 0.0
        self.llm_calls = 0
        self.ranking_time = 0.0
        self.ce_ranking_time = 0.0
        self.llm_inputs = 0
        self.llm_outputs = 0
        self.failed: List[Tuple[str, str]] = []
        self.hits_at_1 = 0
        self.hits_at_5 = 0
        self.hits_at_10 = 0
        self.hits_at_20 = 0
        self.hits_at_1_ce = 0
        self.hits_at_5_ce = 0
        self.hits_at_10_ce = 0
        self.hits_at_20_ce = 0
        self.select = 0
        self.no_select = 0
        self.no_no = 0
        
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
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "hits_at_1": self.hits_at_1,
                "hits_at_5": self.hits_at_5,
                "hits_at_1_ce": self.hits_at_1_ce,
                "hits_at_5_ce": self.hits_at_5_ce,
                "select": self.select,
                "no_select": self.no_select,
                "no_no": self.no_no,
                "returned_paths": (self.tp + self.fp),
                "predictions_per_query": (self.tp + self.fp) / self.tested_paths if self.tested_paths > 0 else 0,
                "time": self.time,
                "sparql_time": self.sparql_time,
                "sparql_calls": self.sparql_calls,
                "llm_time": self.llm_time,
                "llm_calls": self.llm_calls,
                "ranking_time": self.ranking_time,
                "ce_ranking_time": self.ce_ranking_time,
                "time_per_path": self.time / self.tested_paths if self.tested_paths > 0 else 0,
                "sparql_time_per_path": self.sparql_time / self.tested_paths if self.tested_paths > 0 else 0,
                "llm_time_per_path": self.llm_time / self.tested_paths if self.tested_paths > 0 else 0,
                "ranking_time_per_path": self.ranking_time / self.tested_paths if self.tested_paths > 0 else 0,
                "ce_ranking_time_per_path": self.ce_ranking_time / self.tested_paths if self.tested_paths > 0 else 0,
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
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "hits_at_1": self.hits_at_1,
                "hits_at_5": self.hits_at_5,
                "hits_at_10": self.hits_at_10,
                "hits_at_20": self.hits_at_20,
                "hits_at_1_ce": self.hits_at_1_ce,
                "hits_at_5_ce": self.hits_at_5_ce,
                "hits_at_10_ce": self.hits_at_10_ce,
                "hits_at_20_ce": self.hits_at_20_ce,
                "returned_paths": (self.tp + self.fp),
                "predictions_per_query": (self.tp + self.fp) / self.tested_paths if self.tested_paths > 0 else 0,
                "time": self.time,
                "ranking_time": self.ranking_time,
                "ce_ranking_time": self.ce_ranking_time,
                "time_per_path": self.time / self.tested_paths if self.tested_paths > 0 else 0,
                "ranking_time_per_path": self.ranking_time / self.tested_paths if self.tested_paths > 0 else 0,
                "ce_ranking_time_per_path": self.ce_ranking_time / self.tested_paths if self.tested_paths > 0 else 0,
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
        
        
def evaluate_and_save_results(question: str, paths: List[KgPath], path_metrics: GraphSearchMetrics, metrics: Metrics, metrics_output_file, results_output_file):    
    metrics.time += path_metrics.time
    metrics.tested_paths += 1
    metrics.sparql_time += path_metrics.sparql_time
    metrics.sparql_calls += path_metrics.sparql_calls
    metrics.llm_time += path_metrics.llm_time
    metrics.llm_calls += path_metrics.llm_calls
    metrics.llm_inputs += path_metrics.llm_inputs
    metrics.llm_outputs += path_metrics.llm_outputs
    paths_tuples_list = [path.get_tuples() for path in paths]
    match_found = any(check_tuples_equality(found_tuples, expected_tuples) for found_tuples in paths_tuples_list)
    # print(f"  Shortest paths found: {shortest_paths_tuples_list}")
    if match_found:
        metrics.tp += 1
        print(f"\t{metrics.name}: ✅ in {path_metrics.time:.4f} seconds")
    else:
        print(f"\t{metrics.name}: ❌ in {path_metrics.time:.4f} seconds")
        metrics.fn += 1
        metrics.failed.append((start_uri, end_uri))
    metrics.fp += len(paths) - (1 if match_found else 0)
    # rank
    start_time = time.time()
    ranked_paths = rank_paths_by_embedding_similarity(question, kg_enum.value, paths)
    end_time = time.time()
    metrics.ranking_time += (end_time - start_time)
    ranked_paths_tuples_list = [path.get_tuples() for path in ranked_paths]
    if match_found:
        match_index = next(i for i, found_tuples in enumerate(ranked_paths_tuples_list) if check_tuples_equality(found_tuples, expected_tuples))
        if match_index == 0:
            metrics.hits_at_1 += 1
        if match_index < 5:
            metrics.hits_at_5 += 1
        if match_index < 10:
            metrics.hits_at_10 += 1
        if match_index < 20:
            metrics.hits_at_20 += 1
    # cross encoder ranking
    start_time = time.time()
    ranked_paths_ce = rank_paths_with_cross_encoder(question, kg_enum.value, paths)
    end_time = time.time()
    metrics.ce_ranking_time += (end_time - start_time)
    ranked_paths_ce_tuples_list = [path.get_tuples() for path in ranked_paths_ce]
    if match_found:
        match_index = next(i for i, found_tuples in enumerate(ranked_paths_ce_tuples_list) if check_tuples_equality(found_tuples, expected_tuples))
        if match_index == 0:
            metrics.hits_at_1_ce += 1
        if match_index < 5:
            metrics.hits_at_5_ce += 1
        if match_index < 10:
            metrics.hits_at_10_ce += 1
        if match_index < 20:
            metrics.hits_at_20_ce += 1
    save_scores(metrics, metrics_output_file)
    
    per_entry_results = {
        "question": question,
        "found": match_found,
        "selected_paths": [path.get_tuples() for path in paths],
    }
    metrics.per_entry_results.append(per_entry_results)
    save_results(metrics.per_entry_results, results_output_file)


if __name__ == "__main__":
    create_console_logger()
    
    TEST_SHORTEST_PATHS = True
    TEST_ALL_PATHS = True
    TEST_ALL_PATHS_WITH_SHORTEST_FALLBACK = True
    TEST_GRAPH_SEARCH_ADHOC = False
    TEST_GRAPH_SEARCH_LLM = True
    TEST_GRAPH_SEARCH_TRIPLES = False
    TEST_GRAPH_SEARCH_VERBALIZED = False
    
    datasets = [
        ('../../../datasets/_graph_search_evaluation/QALD-9_graph_exploration_evaluation.jsonl', KnowledgeGraphs.DBPEDIA10, "dbpedia10"),
        ('../../../datasets/_graph_search_evaluation/QALD-10_graph_exploration_evaluation.jsonl', KnowledgeGraphs.WIKIDATA, "wikidata"),
        ('../../../datasets/_graph_search_evaluation/WebQSP_graph_exploration_evaluation.jsonl', KnowledgeGraphs.FREEBASE, "freebase"),
        ('../../../datasets/_graph_search_evaluation/LC-QuAD_graph_exploration_evaluation.jsonl', KnowledgeGraphs.DBPEDIA, "dbpedia2016"),
    ]
    
    for dataset_path, kg_enum, graph_string in datasets:
        print(f"Setting up GraphDB for graph: {graph_string}")
        setup_graphdb(kg_enum.value.endpoint)
        
        print(f"Setting up graph tool for graph: {graph_string}")
        setup_graph_tool(graph_string)
        
        print(f"Setting up knowledge graph: {kg_enum.name}")
        kg_enum.load(os.path.join(CONFIG().get('index_dir'), graph_string))
        
        print(f"Loading dataset from: {dataset_path}")
        data = load_dataset(dataset_path)
        
        # for item in data:
            # print(f"Start: {item['start_uri']}, End: {item['end_uri']}, Path Type: {item['path_type']}, Tuples: {item['tuples']}")
        
        print(f"Initializing GraphSearchManager for graph: {graph_string}")
        graph_search_manager = GraphSearchManager(kg_enum, ProviderFactory.create_from_config(CONFIG()))
        
        total_paths = len(data)
        tested_paths = 0
        
        class_to_class_skipped = 0
        unknown_skipped = 0
        invalid_connections = 0
        
        metrics_shortest_paths = Metrics("shortest_paths")
        shortest_paths_metrics_output_file = dataset_path.replace(".jsonl", f"_shortest_paths_metrics.json")
        shortest_paths_output_file = dataset_path.replace(".jsonl", f"_shortest_paths_results.json")
        
        metrics_all_paths = Metrics("all_paths")
        all_paths_metrics_output_file = dataset_path.replace(".jsonl", f"_all_paths_metrics.json")
        all_paths_output_file = dataset_path.replace(".jsonl", f"_all_paths_results.json")
        
        metrics_all_paths_fb = Metrics("all_paths_fallback")
        all_paths_fb_metrics_output_file = dataset_path.replace(".jsonl", f"_all_paths_fallback_metrics.json")
        all_paths_fb_output_file = dataset_path.replace(".jsonl", f"_all_paths_fallback_results.json")
        
        metrics_graph_search_adhoc = Metrics("graph_search_adhoc")
        graph_search_adhoc_metrics_output_file = dataset_path.replace(".jsonl", f"_graph_search_adhoc_metrics.json")
        graph_search_adhoc_output_file = dataset_path.replace(".jsonl", f"_graph_search_adhoc_results.json")
        
        metrics_graph_search_llm = Metrics("graph_search_llm")
        graph_search_llm_metrics_output_file = dataset_path.replace(".jsonl", f"_graph_search_llm_metrics.json")
        graph_search_llm_output_file = dataset_path.replace(".jsonl", f"_graph_search_llm_results.json")
        
        metrics_graph_search_triples = Metrics("graph_search_triples")
        graph_search_triples_metrics_output_file = dataset_path.replace(".jsonl", f"_graph_search_triples_metrics.json")
        graph_search_triples_output_file = dataset_path.replace(".jsonl", f"_graph_search_triples_results.json")
        
        metrics_graph_search_verbalized = Metrics("graph_search_verbalized")
        graph_search_verbalized_metrics_output_file = dataset_path.replace(".jsonl", f"_graph_search_verbalized_metrics.json")
        graph_search_verbalized_output_file = dataset_path.replace(".jsonl", f"_graph_search_verbalized_results.json")
        
        print(f"Evaluating {total_paths} paths...")
        for idx, item in tqdm.tqdm(enumerate(data), desc="Evaluating paths", total=total_paths):
            question = item["question"]
            start_uri = item["start_uri"]
            end_uri = item["end_uri"]
            expected_tuples = item["tuples"]
            
            print(f"Expecting path from {start_uri} to {end_uri} with tuples: {expected_tuples}")
            
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
            
            if kg_enum.value.is_class(start_uri) and kg_enum.value.is_class(end_uri):
                print(f"Skipping evaluation for class to class path: {start_uri} to {end_uri}")
                class_to_class_skipped += 1
                continue
            if kg_enum.value.is_class(start_uri) == None or kg_enum.value.is_class(end_uri) == None:
                print(f"Could not determine if start or end node is a class: {start_uri}, {end_uri}")
                unknown_skipped += 1
                continue
            
            tested_paths += 1
            
            if TEST_SHORTEST_PATHS:
                shortest_paths, shortest_paths_metrics = graph_search_manager._shortest_paths(start_uri, end_uri)
                evaluate_and_save_results(question, shortest_paths, shortest_paths_metrics, metrics_shortest_paths, shortest_paths_metrics_output_file, shortest_paths_output_file)
                # print("Shortest paths found:")
                # for path in shortest_paths:
                #     highlight = False
                #     if check_tuples_equality(path.get_tuples(), expected_tuples):
                #         highlight = True
                #         print("\033[32m", end="")
                #     print(path.get_path_triples_description(readable=True, kg=kg_enum.value, show_specificity=True))
                #     if highlight:
                #         print("\033[0m", end="")
                
            if TEST_ALL_PATHS:
                all_paths, all_paths_metrics = graph_search_manager._all_paths(start_uri, end_uri, filter_predicates=False, additional_hops=1)
                evaluate_and_save_results(question, all_paths, all_paths_metrics, metrics_all_paths, all_paths_metrics_output_file, all_paths_output_file)
                # print("All paths found:")
                # for path in all_paths:
                #     highlight = False
                #     if check_tuples_equality(path.get_tuples(), expected_tuples):
                #         highlight = True
                #         print("\033[32m", end="")
                #     print(path.get_path_triples_description(readable=True, kg=kg_enum.value, show_specificity=True))
                #     if highlight:
                #         print("\033[0m", end="")
            
            if TEST_ALL_PATHS_WITH_SHORTEST_FALLBACK and TEST_SHORTEST_PATHS and TEST_ALL_PATHS:
                all_paths : List[KgPath] = all_paths if len(all_paths) > 0 else shortest_paths
                all_paths_fb_metrics = all_paths_metrics
                evaluate_and_save_results(question, all_paths, all_paths_fb_metrics, metrics_all_paths_fb, all_paths_fb_metrics_output_file, all_paths_fb_output_file)
                
            if TEST_GRAPH_SEARCH_ADHOC:
                graph_paths, graph_metrics = graph_search_manager.graph_search(question, start_uri, end_uri, 
                                                                               representation=GraphSearchManager.RepresentationType.ADHOC,
                                                                               prune=True,
                                                                               explain=False)
                metrics_graph_search_adhoc.no_no += graph_metrics.no_no
                metrics_graph_search_adhoc.no_select += graph_metrics.no_select
                metrics_graph_search_adhoc.select += graph_metrics.select
                evaluate_and_save_results(question, graph_paths, graph_metrics, metrics_graph_search_adhoc, graph_search_adhoc_metrics_output_file, graph_search_adhoc_output_file)
                
            if TEST_GRAPH_SEARCH_LLM:
                graph_paths, graph_metrics = graph_search_manager.graph_search(question, start_uri, end_uri, 
                                                                               representation=GraphSearchManager.RepresentationType.LLM,
                                                                               prune=True,
                                                                               explain=False,
                                                                               focus_on_recall=True)
                metrics_graph_search_llm.no_no += graph_metrics.no_no
                metrics_graph_search_llm.no_select += graph_metrics.no_select
                metrics_graph_search_llm.select += graph_metrics.select
                evaluate_and_save_results(question, graph_paths, graph_metrics, metrics_graph_search_llm, graph_search_llm_metrics_output_file, graph_search_llm_output_file)
                # print("LLM selected paths:")
                # for path in graph_paths:
                #     highlight = False
                #     if check_tuples_equality(path.get_tuples(), expected_tuples):
                #         highlight = True
                #         print("\033[32m", end="")
                #     print(path.get_path_triples_description(readable=True, kg=kg_enum.value, show_specificity=True))
                #     if highlight:
                #         print("\033[0m", end="")
                
            if TEST_GRAPH_SEARCH_TRIPLES:
                graph_paths, graph_metrics = graph_search_manager.graph_search(question, start_uri, end_uri, 
                                                                               representation=GraphSearchManager.RepresentationType.TRIPLES,
                                                                               prune=True,
                                                                               explain=False)
                metrics_graph_search_triples.no_no += graph_metrics.no_no
                metrics_graph_search_triples.no_select += graph_metrics.no_select
                metrics_graph_search_triples.select += graph_metrics.select
                evaluate_and_save_results(question, graph_paths, graph_metrics, metrics_graph_search_triples, graph_search_triples_metrics_output_file, graph_search_triples_output_file)
                
            if TEST_GRAPH_SEARCH_VERBALIZED:
                graph_paths, graph_metrics = graph_search_manager.graph_search(question, start_uri, end_uri, 
                                                                               representation=GraphSearchManager.RepresentationType.VERBALIZATION,
                                                                               prune=True,
                                                                               explain=False)
                metrics_graph_search_verbalized.no_no += graph_metrics.no_no
                metrics_graph_search_verbalized.no_select += graph_metrics.no_select
                metrics_graph_search_verbalized.select += graph_metrics.select
                evaluate_and_save_results(question, graph_paths, graph_metrics, metrics_graph_search_verbalized, graph_search_verbalized_metrics_output_file, graph_search_verbalized_output_file)
                
            # exit(1)
                

        shortest_found, shortest_precision, shortest_recall, shortest_f1 = metrics_shortest_paths.calculate_metrics()
        all_found, all_precision, all_recall, all_f1 = metrics_all_paths.calculate_metrics()
        all_fb_found, all_fb_precision, all_fb_recall, all_fb_f1 = metrics_all_paths_fb.calculate_metrics()
        
        complete_results = {
            "total_paths": total_paths,
            "tested_paths": tested_paths,
            "class_to_class_skipped": class_to_class_skipped,
            "unknown_skipped": unknown_skipped,
            "invalid_connections": invalid_connections,
            "shortest_paths": metrics_shortest_paths.dict(),
            "all_paths": metrics_all_paths.dict(),
            "all_paths_with_shortest_fallback": metrics_all_paths_fb.dict(),
        }
        
        with open(dataset_path.replace(".jsonl", f"_complete_results.json"), "w") as file:
            json.dump(complete_results, file, indent=4)
        
        # -----------------
        # ----- TESTS -----
        # -----------------
        
        # tuple_list_a = [
        #     ("?x", "http://example.org/prop1", "http://example.org/entity1"),
        #     ("http://example.org/entity2", "http://example.org/prop2", "?x"),
        # ]
        
        # tuple_list_b = [
        #     ("?y", "http://example.org/prop2", "http://example.org/entity2"),
        #     ("http://example.org/entity1", "http://example.org/prop1", "?y"),
        # ]
        
        # tuple_list_c = [
        #     ("http://example.org/entity2", "http://example.org/prop2", "?y"),
        #     ("?y", "http://example.org/prop1", "http://example.org/entity1"),
        # ]
        
        # tuple_list_d = [
        #     ("?x", "http://example.org/prop1", "http://example.org/entity1"),
        #     ("http://example.org/entity2", "http://example.org/prop2", "?y"),
        # ]
        
        # print("Tuple lists equal (ignoring variable names):", check_tuples_equality(tuple_list_a, tuple_list_b))
        
        # print("Tuple lists equal (ignoring variable names):", check_tuples_equality(tuple_list_a, tuple_list_c))
        
        # print("Tuple lists equal (ignoring variable names):", check_tuples_equality(tuple_list_a, tuple_list_d))
        
        # tuple_list_a = [
        #     ("?y", "http://example.org/prop1", "http://example.org/entity1"),
        #     ("http://example.org/entity2", "http://example.org/prop2", "?y"),
        #     ("http://example.org/entity3", "http://example.org/prop3", "http://example.org/entity4"),
        #     ("http://example.org/entity5", "http://example.org/prop4", "?x"),
        # ]
        
        # tuple_list_b = [
        #     ("?y", "http://example.org/prop1", "http://example.org/entity1"),
        #     ("http://example.org/entity2", "http://example.org/prop2", "?y"),
        #     ("http://example.org/entity3", "http://example.org/prop3", "http://example.org/entity4"),
        #     ("http://example.org/entity5", "http://example.org/prop4", "?z"),
        # ]
        
        # tuple_list_c = [
        #     ("?y", "http://example.org/prop1", "http://example.org/entity1"),
        #     ("http://example.org/entity2", "http://example.org/prop2", "?z"),
        #     ("http://example.org/entity3", "http://example.org/prop3", "http://example.org/entity4"),
        #     ("http://example.org/entity5", "http://example.org/prop4", "?z"),
        # ]
        
        # print("Tuple lists equal (ignoring variable names):", check_tuples_equality(tuple_list_a, tuple_list_b))
        
        # print("Tuple lists equal (ignoring variable names):", check_tuples_equality(tuple_list_a, tuple_list_c))