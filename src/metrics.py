class KgaqaTracker:
    def __init__(self):
        # ---------------------------
        # ----- Dataset Metrics -----
        # ---------------------------
        self._total = 0 # queries that were processed and counted
        self._invalid_gold_queries = 0
        self._empty_gold_queries = 0
        # -------------------------------
        # ----- Performance Metrics -----
        # -------------------------------
        # Simple Query Generator
        self._total_tp = 0
        self._total_fp = 0
        self._total_fn = 0
        self._total_macro_f1 = 0
        self._total_macro_precision = 0
        self._total_macro_recall = 0
        self._total_hits_at_1 = 0
        self._exact_match = 0
        # ICL Query Generator
        self._icl_total_tp = 0
        self._icl_total_fp = 0
        self._icl_total_fn = 0
        self._icl_total_macro_f1 = 0
        self._icl_total_macro_precision = 0
        self._icl_total_macro_recall = 0
        self._icl_total_hits_at_1 = 0
        self._icl_exact_match = 0
        # -------------------
        # ----- General -----
        # -------------------
        self._total_questions = 0           # total number of questions (they might not all be counted, because they might have invalid gold queries)
        self._llm_time = 0.0                # total time taken by the LLM to generate answers
        self._llm_calls = 0                 # total number of LLM calls
        self._llm_inputs = 0                # total number of LLM input tokens
        self._llm_outputs = 0               # total number of LLM output tokens
        self._llm_tokens = 0                # total number of LLM tokens (input + output)
        self._embed_time = 0.0              # total time taken for embedding generation
        self._embed_calls = 0               # total number of embedding calls
        self._sparql_execs = 0              # total number of SPARQL executions
        self._sparql_time = 0.0             # total time taken for SPARQL executions
        self._ri_time = 0.0                 # total time taken for relation identification
        self._pe_time = 0.0                 # total time taken for path extraction
        self._qg_zero_shot_time = 0.0       # total time taken for query generation
        self._qg_icl_time = 0.0             # total time taken for query generation with in-context learning
        # -------------------------------
        # ----- Relation Identifier -----
        # -------------------------------
        self._ri_valid_generations = 0      # how many questions received valid generations
        self._ri_invalid_generations = 0    # how many questions received invalid generations
        self._ri_no_generations = 0         # how many questions received no generations
        self._ri_total_trials = 0
        self._ri_total_time = 0.0
        self._ri_prompt_llm_call = 0         # how many times the LLM was called for relation identification
        self._ri_prompt_llm_time = 0.0       # how much time was spent on the LLM for relation identification
        self._ri_prompt_verbalization_call = 0  # how many times the LLM was called for verbalization
        self._ri_prompt_verbalization_time = 0.0  # how much time was spent on the LLM for verbalization
        # --------------------------
        # ----- Path Extractor -----
        # --------------------------
        self._pe_no_triples_for_connection = 0      # how many connections had no triples found
        self._pe_no_triples_for_property_path = 0   # how many connections had no triples for property path
        self._pe_known_to_known = 0                 # how many connections were known to known
        self._pe_unknown_goal = 0
        self._pe_unknown_start = 0
        self._pe_shortest_calls = 0                 # how many connections were found with the shortest path
        self._pe_shortest_path_time = 0.0
        self._pe_all_paths_calls = 0                # how many connections were found with all paths
        self._pe_all_paths_time = 0.0
        self._pe_graph_search_calls = 0                # how many connections were found with graph search
        self._pe_graph_search_time = 0.0
        self._pe_neighborhood_calls = 0                # how many connections were found with neighborhood search
        self._pe_neighborhood_time = 0.0
        self._pe_neighborhood_predicate_selection_cache_hits = 0
        self._pe_neighborhood_predicate_selection_cache_misses = 0
        self._pe_property_path_to_triples_calls = 0
        self._pe_property_path_to_triples_time = 0.0
        self._pe_prompt_inclusion_time = 0.0
        self._pe_prompt_inclusion_calls = 0
        self._pe_prompt_grounding_time = 0.0
        self._pe_prompt_grounding_calls = 0
        self._pe_prompt_neighborhood_time = 0.0
        self._pe_prompt_neighborhood_calls = 0
        # ---------------------------
        # ----- Query Generator -----
        # ---------------------------
        self._qg_prompt_query_gen_zero_shot_calls = 0
        self._qg_prompt_query_gen_zero_shot_time = 0.0
        self._qg_prompt_query_gen_icl_calls = 0
        self._qg_prompt_query_gen_icl_time = 0.0
        # ------------------
        # ----- SPARQL -----
        # ------------------
        self._is_class_calls = 0
        self._is_class_time = 0.0
        self._is_entity_calls = 0
        self._is_entity_time = 0.0
        self._get_types_for_node_calls = 0
        self._get_types_for_node_time = 0.0
        self._get_get_all_paths_from_to_calls = 0
        self._get_get_all_paths_from_to_time = 0.0
        self._get_shortest_path_from_to_calls = 0
        self._get_shortest_path_from_to_time = 0.0
        self._get_distinct_predicates_for_class_calls = 0
        self._get_distinct_predicates_for_class_time = 0.0
        self._get_distinct_predicates_for_class_hits = 0
        self._get_distinct_predicates_for_class_hits_time = 0.0
        self._get_distinct_predicates_for_class_misses = 0
        self._get_distinct_predicates_for_entity_calls = 0
        self._get_distinct_predicates_for_entity_time = 0.0
        self._get_object_for_subject_predicate_calls = 0
        self._get_object_for_subject_predicate_time = 0.0
        self._get_subject_from_predicate_object_calls = 0
        self._get_subject_from_predicate_object_time = 0.0
        self._are_triples_valid_calls = 0
        self._are_triples_valid_time = 0.0
        self._are_triples_valid_avoided = 0
        self._has_no_value_connection_calls = 0
        self._has_no_value_connection_time = 0.0
        self._replace_types_for_triples_calls = 0
        self._replace_types_for_triples_time = 0.0
        self._triples_popularity_calls = 0
        self._triples_popularity_time = 0.0
        self._triples_results_calls = 0
        self._triples_results_time = 0.0
        self._uri_to_uril_calls_misses = 0
        self._uri_to_uril_time_misses = 0.0
        self._uri_to_uril_calls_hits = 0
        self._uri_to_uril_time_hits = 0.0

        
    def get_metrics(self):
        return {
            "top_metrics 0-shot": {
                "f-score": self._total_macro_f1/ self._total if self._total > 0 else 0.0,
                "hits@1": self._total_hits_at_1 / self._total if self._total > 0 else 0.0,
                "accuracy": self._exact_match / self._total if self._total > 0 else 0.0,
                "time_per_question": (self._ri_time + self._pe_time + self._qg_zero_shot_time) / self._total_questions if self._total_questions > 0 else 0.0,
                "average_llm_calls": (self._llm_calls - self._qg_prompt_query_gen_icl_calls) / self._total_questions if self._total_questions > 0 else 0.0,
            },
            "top_metrics ICL": {
                "f-score": self._icl_total_macro_f1 / self._total if self._total > 0 else 0.0,
                "hits@1": self._icl_total_hits_at_1 / self._total if self._total > 0 else 0.0,
                "accuracy": self._icl_exact_match / self._total if self._total > 0 else 0.0,
                "time_per_question": (self._ri_time + self._pe_time + self._qg_icl_time) / self._total_questions if self._total_questions > 0 else 0.0,
                "average_llm_calls": (self._llm_calls - self._qg_prompt_query_gen_zero_shot_calls) / self._total_questions if self._total_questions > 0 else 0.0,
            },
            "dataset_metrics" : {
                "total_questions": self._total_questions,
                "total_valid": self._total,
                "invalid_gold_queries": self._invalid_gold_queries,
                "empty_gold_queries": self._empty_gold_queries
            },
            "performance_metrics": {
                "total_tp": self._total_tp,
                "total_fp": self._total_fp,
                "total_fn": self._total_fn,
                "total_macro_f1": self._total_macro_f1,
                "total_macro_precision": self._total_macro_precision,
                "total_macro_recall": self._total_macro_recall,
                "total_hits_at_1": self._total_hits_at_1,
                "exact_match": self._exact_match,
                "icl_total_tp": self._icl_total_tp,
                "icl_total_fp": self._icl_total_fp,
                "icl_total_fn": self._icl_total_fn,
                "icl_total_macro_f1": self._icl_total_macro_f1,
                "icl_total_macro_precision": self._icl_total_macro_precision,
                "icl_total_macro_recall": self._icl_total_macro_recall,
                "icl_total_hits_at_1": self._icl_total_hits_at_1,
                "icl_exact_match": self._icl_exact_match
            },
            "general_metrics": {
                "llm_time": self._llm_time,
                "llm_calls": self._llm_calls,
                "llm_inputs": self._llm_inputs,
                "llm_outputs": self._llm_outputs,
                "llm_tokens": self._llm_tokens,
                "embed_time": self._embed_time,
                "embed_calls": self._embed_calls,
                "sparql_execs": self._sparql_execs,
                "sparql_time": self._sparql_time,
                "ri_time": self._ri_time,
                "pe_time": self._pe_time,
                "qg_zero_shot_time": self._qg_zero_shot_time,
                "qg_icl_time": self._qg_icl_time
            },
            "relation_identifier_metrics": {
                "ri_valid_generations": self._ri_valid_generations,
                "ri_invalid_generations": self._ri_invalid_generations,
                "ri_no_generations": self._ri_no_generations,
                "ri_total_trials": self._ri_total_trials,
                "ri_total_time": self._ri_total_time,
                "ri_prompt_llm_call": self._ri_prompt_llm_call,
                "ri_prompt_llm_time": self._ri_prompt_llm_time,
                "ri_prompt_verbalization_call": self._ri_prompt_verbalization_call,
                "ri_prompt_verbalization_time": self._ri_prompt_verbalization_time
            },
            "path_extractor_metrics": {
                "pe_no_triples_for_connection": self._pe_no_triples_for_connection,
                "pe_no_triples_for_property_path": self._pe_no_triples_for_property_path,
                "pe_known_to_known": self._pe_known_to_known,
                "pe_unknown_goal": self._pe_unknown_goal,
                "pe_unknown_start": self._pe_unknown_start,
                "pe_shortest_calls": self._pe_shortest_calls,
                "pe_shortest_path_time": self._pe_shortest_path_time,
                "pe_all_paths_calls": self._pe_all_paths_calls,
                "pe_all_paths_time": self._pe_all_paths_time,
                "pe_graph_search_calls": self._pe_graph_search_calls,
                "pe_graph_search_time": self._pe_graph_search_time,
                "pe_neighborhood_calls": self._pe_neighborhood_calls,
                "pe_neighborhood_time": self._pe_neighborhood_time,
                "pe_neighborhood_predicate_selection_cache_hits": self._pe_neighborhood_predicate_selection_cache_hits,
                "pe_neighborhood_predicate_selection_cache_misses": self._pe_neighborhood_predicate_selection_cache_misses,
                "pe_property_path_to_triples_calls": self._pe_property_path_to_triples_calls,
                "pe_property_path_to_triples_time": self._pe_property_path_to_triples_time,
                "pe_prompt_inclusion_time": self._pe_prompt_inclusion_time,
                "pe_prompt_inclusion_calls": self._pe_prompt_inclusion_calls,
                "pe_prompt_grounding_time": self._pe_prompt_grounding_time,
                "pe_prompt_grounding_calls": self._pe_prompt_grounding_calls,
                "pe_prompt_neighborhood_time": self._pe_prompt_neighborhood_time,
                "pe_prompt_neighborhood_calls": self._pe_prompt_neighborhood_calls
            },
            "query_generator_metrics": {
                "qg_prompt_query_gen_zero_shot_calls": self._qg_prompt_query_gen_zero_shot_calls,
                "qg_prompt_query_gen_zero_shot_time": self._qg_prompt_query_gen_zero_shot_time,
                "qg_prompt_query_gen_icl_calls": self._qg_prompt_query_gen_icl_calls,
                "qg_prompt_query_gen_icl_time": self._qg_prompt_query_gen_icl_time
            },
            "sparql_metrics": {
                "is_class_calls": self._is_class_calls,
                "is_class_time": self._is_class_time,
                "is_entity_calls": self._is_entity_calls,
                "is_entity_time": self._is_entity_time,
                "get_types_for_node_calls": self._get_types_for_node_calls,
                "get_types_for_node_time": self._get_types_for_node_time,
                "get_get_all_paths_from_to_calls": self._get_get_all_paths_from_to_calls,
                "get_get_all_paths_from_to_time": self._get_get_all_paths_from_to_time,
                "get_shortest_path_from_to_calls": self._get_shortest_path_from_to_calls,
                "get_shortest_path_from_to_time": self._get_shortest_path_from_to_time,
                "get_distinct_predicates_for_class_calls": self._get_distinct_predicates_for_class_calls,
                "get_distinct_predicates_for_class_time": self._get_distinct_predicates_for_class_time,
                "get_distinct_predicates_for_class_hits": self._get_distinct_predicates_for_class_hits,
                "get_distinct_predicates_for_class_hits_time": self._get_distinct_predicates_for_class_hits_time,
                "get_distinct_predicates_for_class_misses": self._get_distinct_predicates_for_class_misses,
                "get_distinct_predicates_for_entity_calls": self._get_distinct_predicates_for_entity_calls,
                "get_distinct_predicates_for_entity_time": self._get_distinct_predicates_for_entity_time,
                "get_object_for_subject_predicate_calls": self._get_object_for_subject_predicate_calls,
                "get_object_for_subject_predicate_time": self._get_object_for_subject_predicate_time,
                "get_subject_from_predicate_object_calls": self._get_subject_from_predicate_object_calls,
                "get_subject_from_predicate_object_time": self._get_subject_from_predicate_object_time,
                "are_triples_valid_calls": self._are_triples_valid_calls,
                "are_triples_valid_time": self._are_triples_valid_time,
                "are_triples_valid_avoided": self._are_triples_valid_avoided,
                "has_no_value_connection_calls": self._has_no_value_connection_calls,
                "has_no_value_connection_time": self._has_no_value_connection_time,
                "replace_types_for_triples_calls": self._replace_types_for_triples_calls,
                "replace_types_for_triples_time": self._replace_types_for_triples_time,
                "triples_popularity_calls": self._triples_popularity_calls,
                "triples_popularity_time": self._triples_popularity_time,
                "triples_results_calls": self._triples_results_calls,
                "triples_results_time": self._triples_results_time,
                "uri_to_uril_calls_misses": self._uri_to_uril_calls_misses,
                "uri_to_uril_time_misses": self._uri_to_uril_time_misses,
                "uri_to_uril_calls_hits": self._uri_to_uril_calls_hits,
                "uri_to_uril_time_hits": self._uri_to_uril_time_hits
            }
        }
        
    def load_from_dict(self, metrics_dict):
        for key, value in metrics_dict.items():
            for key2, value2 in value.items():
                if key2 == "total_valid":
                    key2 = "total"
                if hasattr(self, f"_{key2}"):
                    setattr(self, f"_{key2}", value2)
                else:
                    print(f"Warning: Metric '{key2}' not found in KgaqaTracker.")
    
    def print(self):
        metrics = self.get_metrics()
        for category, values in metrics.items():
            print(f"{category}:")
            max_key_length = max(len(key) for key in values.keys())
            for key, value in values.items():
                print(f"\t{key:<{max_key_length}} : {value}")
        print("\n")
        
tracker = None

def get_kgaqa_tracker(new=False) -> KgaqaTracker:
    global tracker
    if new or tracker is None:
        tracker = KgaqaTracker()
    return tracker

def get_kgaqa_tracker_from_dict(metrics_dict: dict) -> KgaqaTracker:
    global tracker
    if tracker is None:
        tracker = KgaqaTracker()
    tracker.load_from_dict(metrics_dict)
    return tracker

class PerformanceMetrics:
    
    def __init__(self, sparql_calls=0, sparql_time=0.0, llm_calls=0, llm_time=0.0, llm_inputs=0, llm_outputs=0):
        self.sparql_calls = sparql_calls
        self.sparql_time = sparql_time
        self.llm_calls = llm_calls
        self.llm_time = llm_time
        self.llm_inputs = llm_inputs
        self.llm_outputs = llm_outputs
    
    def to_dict(self):
        return {
            "SPARQL_CALLS": self.sparql_calls,
            "SPARQL_TIME": self.sparql_time,
            "LLM_CALLS": self.llm_calls,
            "LLM_TIME": self.llm_time,
            "LLM_INPUTS": self.llm_inputs,
            "LLM_OUTPUTS": self.llm_outputs
        }
        
import time

class Snapshot:
    def __init__(self, time, llm_time, input_tokens, output_tokens, total_tokens, calls, sparql_time, sparql_calls):
        self.time = time
        # LLMs
        self.llm_time = llm_time
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.calls = calls
        # SPARQL
        self.sparql_time = sparql_time
        self.sparql_calls = sparql_calls

def snapshot_metrics(tracker: KgaqaTracker):
    ttime = time.time()
    llm_time = tracker._llm_time
    input_tokens = tracker._llm_inputs
    output_tokens = tracker._llm_outputs
    total_tokens = tracker._llm_tokens
    calls = tracker._llm_calls
    sparql_time = tracker._sparql_time
    sparql_calls = tracker._sparql_execs
    return Snapshot(ttime, llm_time, input_tokens, output_tokens, total_tokens, calls, sparql_time, sparql_calls)

def change_between_snapshots(snapshot1: Snapshot, snapshot2: Snapshot) -> PerformanceMetrics:
    return PerformanceMetrics(
        sparql_calls = snapshot2.sparql_calls - snapshot1.sparql_calls,
        sparql_time = snapshot2.sparql_time - snapshot1.sparql_time,
        llm_calls = snapshot2.calls - snapshot1.calls,
        llm_time = snapshot2.llm_time - snapshot1.llm_time,
        llm_inputs = snapshot2.input_tokens - snapshot1.input_tokens,
        llm_outputs = snapshot2.output_tokens - snapshot1.output_tokens
    )
    
def change_since_snapshot(snapshot: Snapshot) -> PerformanceMetrics:
    current_snapshot = snapshot_metrics(get_kgaqa_tracker())
    return change_between_snapshots(snapshot, current_snapshot)