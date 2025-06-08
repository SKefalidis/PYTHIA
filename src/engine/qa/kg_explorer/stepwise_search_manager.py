from functools import lru_cache, partial
import re
import time

from dataclasses import dataclass
from textwrap import dedent
from typing import List, Tuple, Set, Dict
from sklearn.metrics.pairwise import cosine_similarity

from src.engine.config import CONFIG
from src.utils import llm_call, is_uri, embed
from src.logging import log, LogComponent, LogLevel
from src.elelem.provider import Provider, ProviderFactory
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.knowledge_graphs.knowledge_graph import KgComponent, KgEntity, PredicateInfo
from src.engine.qa.kg_explorer.stepwise_search_path import *


class StepwiseSearchMetrics:
    def __init__(self):
        self.time = 0.0
        self.sparql_calls = 0
        self.sparql_time = 0.0
        self.llm_calls = 0
        self.llm_time = 0.0
        self.llm_inputs = 0
        self.llm_outputs = 0
        self.steps = 0
        

class StepwiseSearchResultEnum:
    FOUND = "FOUND"
    CANCEL = "CANCEL"  
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    DEAD_END_REACHED = "DEAD_END_REACHED"


@dataclass
class SearchLlmResponse:
    # expand
    expand: bool
    selected_indices: List[str]
    value_scores: List[float]
    alternative_value_scores: List[float]
    # found
    found: bool
    found_indices: List[str]
    # cancel
    cancel: bool
    # invalid
    invalid: bool


class SearchState:

    def __init__(self, question: str, start_uri: str, start_label: str, window_size: int = 50):
        self.question = question
        self.start_uri = start_uri
        
        self.frontier: List[StepwiseSearchNode] = []
        self.frontier.append(StepwiseSearchNode(start_uri, start_label, None, question))
        
        self.frontier_weights: List[float] = []
        self.frontier_weights.append(1.0) # initial weight
        
        self.visited: Set[StepwiseSearchNode] = set()
        self.previous_frontiers: List[Tuple[StepwiseSearchNode, float]] = [] # list of (StepwiseSearchNode, value)
        
        self.steps_taken: int = 0
        self.window_size: int = window_size
        
    @property
    def visited_uris(self) -> Set[str]:
        uris = set()
        for node in self.visited:
            for v in node.value:
                uris.add(v)
        return uris


class StepwiseSearchManager:
    
    def __init__(self, kg: KnowledgeGraphs, llm: str):
        self.kg: KnowledgeGraph = kg.value
        self.llm = llm
        
    def search(self, question: str, start_uri: str, max_steps: int = 5, enable_backtracking = False, enable_limit = False, return_partial_results = False, window_size: int = 50) -> Tuple[StepwiseSearchResultEnum, List[StepwiseSearchPath], StepwiseSearchMetrics]:
        self.enable_backtracking = enable_backtracking
        self.enable_limit = enable_limit
        self.return_partial_results = return_partial_results
        
        self.metrics = StepwiseSearchMetrics()
        start_time = time.time()
        state = SearchState(question, start_uri, self.kg.get_kg_component(start_uri).label if self.kg.get_kg_component(start_uri) else self.kg.get_label_from_uri(start_uri), window_size=window_size)
        for step_num in range(max_steps):
            self.metrics.steps += 1
            print(f"--- Step {step_num + 1} ---")
            llm_response, new_nodes = self.step(state)
            
            # Handle case where terminal nodes are selected for expansion.
            activate_improper_selection_handling = False
            count_literal_nodes = 0
            for node in new_nodes:
                if node.node_type == StepwiseSearchNodeType.LITERAL:
                    print("LLM selected to expand to a terminal literal node:", node.value)
                    count_literal_nodes += 1
            if count_literal_nodes > len(new_nodes) / 2:
                print(f"Majority of new nodes ({count_literal_nodes} out of {len(new_nodes)}) are terminal literal nodes. Ending search.")
                activate_improper_selection_handling = True
            
            if llm_response.found or activate_improper_selection_handling:
                paths = [StepwiseSearchPath(self.kg, node) for node in new_nodes]
                self.metrics.time += time.time() - start_time
                return StepwiseSearchResultEnum.FOUND, paths, self.metrics
            elif llm_response.cancel:
                self.metrics.time += time.time() - start_time
                partial_paths = self._get_partial_results(state) if self.return_partial_results else []
                return StepwiseSearchResultEnum.CANCEL, partial_paths, self.metrics
            elif llm_response.invalid:
                self.metrics.time += time.time() - start_time
                partial_paths = self._get_partial_results(state) if self.return_partial_results else []
                return StepwiseSearchResultEnum.CANCEL, partial_paths, self.metrics
            else:
                if len(new_nodes) == 0:
                    print("No new nodes were added to the frontier. Ending search.")
                    self.metrics.time += time.time() - start_time
                    partial_paths = self._get_partial_results(state) if self.return_partial_results else []
                    return StepwiseSearchResultEnum.DEAD_END_REACHED, partial_paths, self.metrics
                else:
                    print(f"Added {len(new_nodes)} new nodes to the frontier. Continuing search.")
                    continue
        self.metrics.time += time.time() - start_time
        partial_paths = self._get_partial_results(state) if self.return_partial_results else []
        return StepwiseSearchResultEnum.MAX_STEPS_REACHED, partial_paths, self.metrics
    
    def step(self, state: SearchState) -> Tuple[SearchLlmResponse, List[StepwiseSearchNode]]:
        # retrieve predicate info for frontier nodes.
        self.find_predicates_of_frontier_nodes(state)
        
        # prune predicates to a manageable number + avoid predicates that lead to visited nodes.
        self.select_frontier_predicates(state, window=50)
        
        # prepare representation for LLM.        
        messages = self.prepare_messages_for_llm(state)
        
        # LLM decides which predicates to follow (def follow).
        trial = 0
        while trial < 3:
            llm_time = time.time()
            try:
                generated, usage = llm_call(self.llm, messages, return_usage=True) 
            except Exception as e:
                print("LLM call failed with error:", e)
                print("Retrying...")
                trial += 1
                continue
            self.metrics.llm_calls += 1
            self.metrics.llm_time += time.time() - llm_time
            self.metrics.llm_inputs += usage.prompt_tokens
            self.metrics.llm_outputs += usage.completion_tokens
            state.steps_taken += 1 # at this point, we have taken a step.      
            print(generated)
            # parse output
            result = self._parse_llm_response(generated, state)
            if result.invalid:
                print("LLM response was invalid. Retrying...")
                trial += 1
            else:
                break
        
        # Handle decisions
        if result.invalid:
            self.mark_frontier_nodes_as_visited(state)
            self.store_frontier_with_values(state, [0.0] * len(state.frontier))
            return result, []
        elif result.cancel:
            self.mark_frontier_nodes_as_visited(state)
            self.store_frontier_with_values(state, [0.0] * len(state.frontier))
            return result, []
        elif result.found:
            found_indices = self._parse_string_indices_list(result.found_indices)
            found_nodes = []
            for branch_idx, predicate_idx in found_indices:
                found_node = state.frontier[branch_idx].generate_successor_by_expanding_frontier_predicate(predicate_idx, self.kg)
                found_nodes.append(found_node)
                print("Found target information at node:", found_node.full_path_representation(readable=True))
            self.mark_frontier_nodes_as_visited(state)
            self.store_frontier_with_values(state, [1.0] * len(state.frontier))
            log(f"LLM decided that the target information has been found at step {state.steps_taken}.", LogComponent.PATH_EXTRACTOR, LogLevel.INFO)
            return result, found_nodes
        else:
            if len(result.selected_indices) == 0:
                raise ValueError("LLM response indicates to expand, but no predicates were selected.")
            if len(result.selected_indices) > 3:
                
                unique_selected_indices = []
                unique_value_scores = []
                for index, weight in zip(result.selected_indices, result.value_scores):
                    if index not in unique_selected_indices:
                        unique_selected_indices.append(index)
                        unique_value_scores.append(weight)
                        
                if len(unique_selected_indices) > 3:
                    print(f"Selected indices: {result.selected_indices}")
                    raise ValueError("LLM response indicates to expand, but more than 3 predicates were selected.")
                else:
                    print(f"LLM selected more than 3 predicates, but after removing duplicates, {len(unique_selected_indices)} unique predicates remain. Proceeding with these.")
                    result.selected_indices = unique_selected_indices
                    result.value_scores = unique_value_scores
            
            selected_indices = self._parse_string_indices_list(result.selected_indices)
            
            new_nodes: List[StepwiseSearchNode] = []
            new_weights: List[float] = []
            for (branch_idx, predicate_idx), weight in zip(selected_indices, result.value_scores):
                print("Selected to expand branch", branch_idx, "with predicate index", predicate_idx)
                new_node = state.frontier[branch_idx].generate_successor_by_expanding_frontier_predicate(predicate_idx, self.kg)
                # TODO: check if objects overlap with existing frontier nodes. If yes, merge paths.
                
                merge = False
                merge_to_idx = -1
                for eidx, existing_node in enumerate(new_nodes):
                    same = True
                    for value in new_node.value:
                        if value not in existing_node.value:
                            same = False
                            break
                    if same:
                        merge = True
                        merge_to_idx = eidx
                        break
                
                if not merge:    
                    new_nodes.append(new_node)
                    new_weights.append(weight)
                    print("Expanding node:", new_node.full_path_representation(readable=True))
                else:
                    print("Branch overlaps with previously expanded branch. Merging with node:", new_nodes[merge_to_idx].value)
                    
                    if new_weights[merge_to_idx] > weight:
                        print(f" Previous weight: {new_weights[merge_to_idx]:.4f}, adding weight: {weight:.4f}")
                        new_weights[merge_to_idx] += weight
                        print(f" New weight: {new_weights[merge_to_idx]:.4f}")
                    else:
                        print(f"Replacing node with new node due to higher weight.")
                        new_nodes[merge_to_idx] = new_node
                        print(f" Previous weight: {new_weights[merge_to_idx]:.4f}, adding weight: {weight:.4f}")
                        new_weights[merge_to_idx] += weight
                        print(f" New weight: {new_weights[merge_to_idx]:.4f}")
            
            self.mark_frontier_nodes_as_visited(state)
            self.store_frontier_with_values(state, result.alternative_value_scores)
            
            for idx, new_node in enumerate(new_nodes):
                state.frontier.append(new_node)
                state.frontier_weights.append(new_weights[idx])
                
                print(" Added to frontier:", new_node.get_textual_representation_of_value(readable=True), " with weight ", new_weights[idx])
        
            return result, new_nodes
        
    def _parse_string_indices_list(self, indices_str_list: List[str]) -> List[Tuple[int,int]]:
        parsed_indices = []
        for index_str in indices_str_list:
            branch_idx = ord(index_str[0]) - ord('A')
            predicate_idx = int(index_str[1:]) # supports more than 26 predicates per node.
            parsed_indices.append((branch_idx, predicate_idx))
        return parsed_indices
    
    def _parse_llm_response(self, response: str, state: SearchState) -> SearchLlmResponse:
        def return_invalid():
            return SearchLlmResponse(
                expand=False,
                selected_indices=[],
                value_scores=[],
                alternative_value_scores=[],
                found=False,
                found_indices=[],
                cancel=False,
                invalid=True)
        llm_response_object = None
        if r"{FOUND}" in response:
            # found answer
            matches = re.findall(r'{[A-Z][\d]+}', response)
            clean_matches = [match.strip("{}") for match in matches]
            if len(clean_matches) == 0:
                print("returning invalid due to no found indices.")
                return return_invalid()
            for match in clean_matches:
                try:
                    indices = self._parse_string_indices_list([match])
                    for branch_idx, idx in indices:
                        if branch_idx >= len(state.frontier):
                            print("returning invalid due to branch index out of range.")
                            return return_invalid()
                        if idx >= len(state.frontier[branch_idx].get_frontier_predicates()):
                            print("returning invalid due to predicate index out of range.")
                            return return_invalid()
                except Exception as e:
                    print("returning invalid due to parsing error in found indices.")
                    print(e)
                    print(match)
                    return return_invalid()
            llm_response_object = SearchLlmResponse(
                expand=False,
                selected_indices=[],
                value_scores=[],
                alternative_value_scores=[],
                found=True,
                found_indices=clean_matches,
                cancel=False,
                invalid=False)
        elif r"{NONE}" in response:
            llm_response_object = SearchLlmResponse(
                expand=False,
                selected_indices=[],
                value_scores=[],
                alternative_value_scores=[],
                found=False,
                found_indices=[],
                cancel=True,
                invalid=False)
        else:
            matches = re.findall(r'{([A-Z][\d]+)-(\d+\.\d+)}', response)
            alternative_value_scores = re.findall(r'\[\d+\.\d+\]', response)
            clean_matches = [match[0] for match in matches]
            if len(clean_matches) == 0:
                print("returning invalid due to no selected indices.")
                return return_invalid()
            for match in clean_matches:
                try:
                    indices = self._parse_string_indices_list([match])
                    for branch_idx, idx in indices:
                        if branch_idx >= len(state.frontier):
                            print("returning invalid due to branch index out of range.")
                            return return_invalid()
                        if idx >= len(state.frontier[branch_idx].get_frontier_predicates()):
                            print("returning invalid due to predicate index out of range.")
                            return return_invalid()
                except Exception as e:
                    print("returning invalid due to parsing error in selected indices.")
                    print(e)
                    print(match)
                    return return_invalid()
            value_scores = [float(match[1]) for match in matches]
            alternative_value_scores = [float(score.strip("[]")) for score in alternative_value_scores] if len(alternative_value_scores) > 0 else []
            llm_response_object = SearchLlmResponse(
                expand=True,
                selected_indices=clean_matches,
                value_scores=value_scores,
                alternative_value_scores=alternative_value_scores,
                found=False,
                found_indices=[],
                cancel=False,
                invalid=False)
        return llm_response_object
    
    def _get_predicate_info_for_node(self, node_uri: str|List[str], filter_literals=False) -> List[PredicateInfo]:
        if isinstance(node_uri, str):
            node_uri = [node_uri]
            
        predicates_per_uri: Dict[str, PredicateInfo] = {}
        predicates_popularity_per_uri: Dict[str, int] = {}
        for node in node_uri:
            if not is_uri(node):
                log(f"Node {node} is not a valid URI.", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
                continue
            kgc: KgComponent|None = self.kg.get_kg_component(node)
            limit = 0
            if self.enable_limit == True:
                limit = 1_000_000
            if kgc is None:
                print("KG component not found for node:", node)
                predicates: List[PredicateInfo] = KgEntity.get_predicates_for_entity(node, self.kg.endpoint, filter_literals=filter_literals, limit=limit)        
            elif kgc.is_entity():
                print("Retrieving predicates for entity node:", node)
                predicates: List[PredicateInfo] = kgc.get_predicates(self.kg.endpoint, filter_literals=filter_literals, limit=limit)
            elif kgc.is_class():
                print("Retrieving predicates for class node:", node)
                predicates: List[PredicateInfo] = kgc.get_own_predicates()
                for predicate in predicates:
                    objects = self.kg.get_object_for_node_predicate_info(node, predicate)
                    predicate.objects = objects
            else:
                print("Node is neither an entity nor a class:", node)
                log(f"Node {node_uri} is neither an entity nor a class", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
                return []
            for predicate in predicates:
                if predicate.uri not in predicates_per_uri:
                    predicates_per_uri[predicate.uri] = predicate
                    predicates_popularity_per_uri[predicate.uri] = 1
                else:
                    # merge objects
                    existing_predicate = predicates_per_uri[predicate.uri]
                    existing_predicate.objects.extend(predicate.objects)
                    existing_predicate.objects = list(set(existing_predicate.objects))
                    existing_predicate.cardinality = len(existing_predicate.objects) 
                    predicates_popularity_per_uri[predicate.uri] += 1
        
        final_predicates = []
        for predicate_uri, predicate in predicates_per_uri.items():
            popularity = predicates_popularity_per_uri[predicate_uri]
            if popularity / len(node_uri) >= 0.4:  # at least 40% of nodes have this predicate
                final_predicates.append(predicate)   
                
        # print(f"Filtered predicates to {len(final_predicates)} out of {len(predicates_per_uri)} based on popularity threshold.")
        # for predicate in final_predicates:
        #     print(f" Predicate: {predicate.get_label(kg=self.kg)} - Objects: {predicate.get_objects_string(readable=True, kg=self.kg)}") 
                            
        return final_predicates
    
    # -----------------------------
    # ----- Frontier Handling -----
    # -----------------------------
    
    def find_predicates_of_frontier_nodes(self, state: SearchState):
        for node in state.frontier:
            if node.node_type == StepwiseSearchNodeType.LITERAL:
                continue
            if len(node.get_predicates()) == 0:
                print("Retrieving predicates for node:", node.value)
                start_time = time.time()
                predicates = self._get_predicate_info_for_node(node.value, filter_literals=False)
                node.set_predicates(predicates)
                print(f"Retrieved {len(predicates)} predicates in {time.time() - start_time:.2f} seconds.")
                
    def select_frontier_predicates(self, state: SearchState, window: int):
        def rank_by_similarity(question, kg, path_verbalizations: List[str]) -> List[int]:
            if len(path_verbalizations) == 0:
                return []
            
            question_embedding = embed(question, is_query=False)
            
            path_embeddings = embed(path_verbalizations, is_query=False)
            
            path_indices = list(range(len(path_verbalizations)))
            path_similarities = cosine_similarity([question_embedding], path_embeddings)[0]
            ranked_paths_indices = [idx for _, _, idx in sorted(zip(path_similarities, path_verbalizations, path_indices), key=lambda x: x[0], reverse=True)]            
            # print("Ranked paths by similarity:")
            # for sim, path in sorted(zip(path_similarities, path_verbalizations), key=lambda x: x[0], reverse=True):
                # print(f"Similarity: {sim:.4f} - Path: {path}")
            
            return ranked_paths_indices
        
        total_weight = sum(state.frontier_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in state.frontier_weights]
        else:
            normalized_weights = [1.0 / len(state.frontier) for _ in state.frontier]
        print("Pruning frontier predicates with window size", window)
        for node, weight in zip(state.frontier, normalized_weights):
            if node.node_type == StepwiseSearchNodeType.LITERAL:
                continue
            start_time = time.time()
            raw_predicates = node.get_predicates()
            print(f"Pruning predicates ({len(raw_predicates)}) for node {node.value} with weight {weight:.4f} (window of {round(window*weight)} predicates)")
            predicates = []
            for predicate in raw_predicates:
                leads_to_visited = False
                new_predicate_objects = []
                for obj in predicate.objects:
                    if obj in state.visited_uris:
                        leads_to_visited = True
                    else:
                        new_predicate_objects.append(obj)
                if leads_to_visited:
                    if len(new_predicate_objects) == 0:
                        print(f" Skipping predicate {predicate.get_label(kg=self.kg)} as it only leads to visited nodes.")
                        continue
                    else:
                        print(f" Pruning objects of predicate {predicate.get_label(kg=self.kg)} to avoid visited nodes.")
                        predicate.objects = new_predicate_objects
                        predicate.cardinality = len(new_predicate_objects)
                predicates.append(predicate)
                
            print(f" After pruning, {len(predicates)} predicates remain.")
            
            verbalizations = []
            for predicate in predicates:
                # verbalization = node.full_path_representation_with_frontier_predicate(predicate, kg=self.kg, readable=True, verbal=True) # might be better to do pure predicate paths, without any other words.
                verbalization = node.predicate_path_representation_with_frontier_predicate(predicate, kg=self.kg, readable=True, verbal=True)
                verbalizations.append(verbalization)
            ranked_indices = rank_by_similarity(state.question, self.kg, verbalizations)
            ranked_predicates = [predicates[i] for i in ranked_indices]
            pruned_predicates = ranked_predicates[:round(window*weight)]
            node.set_frontier_predicates(pruned_predicates)
            print(f"Pruned to {len(pruned_predicates)} predicates in {time.time() - start_time:.2f} seconds.")
            
    def mark_frontier_nodes_as_visited(self, state: SearchState):
        for node in state.frontier:
            state.visited.add(node)
            
    def store_frontier_with_values(self, state: SearchState, values: List[float]):
        for node, value in zip(state.frontier, values):
            state.previous_frontiers.append((node, value))
        state.frontier = []
        state.frontier_weights = []

    def _get_partial_results(self, state: SearchState, top_k: int = 3, length_boost_per_hop: float = 0.15) -> List[StepwiseSearchPath]:
        if len(state.previous_frontiers) == 0:
            return []

        def path_length(node: StepwiseSearchNode) -> int:
            length = 0
            current = node
            while current.previous is not None:
                length += 1
                current = current.previous.start_node
            return length

        def effective_score(node: StepwiseSearchNode, base_score: float) -> float:
            length = path_length(node)
            return base_score * (1.0 + length_boost_per_hop * length)

        scored_frontiers = []
        for node, value in state.previous_frontiers:
            scored_frontiers.append((node, effective_score(node, value)))

        sorted_frontiers = sorted(scored_frontiers, key=lambda item: item[1], reverse=True)
        top_nodes: List[StepwiseSearchNode] = []
        seen_nodes: Set[StepwiseSearchNode] = set()
        for node, _ in sorted_frontiers:
            if node in seen_nodes:
                continue
            top_nodes.append(node)
            seen_nodes.add(node)
            if len(top_nodes) == top_k:
                break
        return [StepwiseSearchPath(self.kg, node) for node in top_nodes]
            
    # -------------------
    # ----- PROMPTS -----
    # -------------------
    
    def prepare_messages_for_llm(self, state: SearchState) -> List[Dict[str, str]]:
        system_prompt = self.generate_system_prompt()
        user_prompt = self.generate_user_prompt(state)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages
    
    def generate_system_prompt(self) -> str:
        prompt = dedent(f"""\
        ### Role
        You are an AI assistant navigating a Knowledge Graph. Your goal is to identify the specific predicates (edges) that lead to the requested **Target Information**.

        ### Objective
        You will receive a set of nodes. Each node has a list of predicates connecting to other values or entities.
        You must analyze the predicates and determine if:
        1. The answer is **Present** (either as a visible text value OR as a specific attribute endpoint).
        2. The answer is **Not Present**, and you need to **Expand** to neighbor nodes to find it.
        3. The answer is **Unreachable** (Give Up).

        ### Critical Instructions

        **1. Handling "Unnamed Entities"**
        The graph contains "Unnamed Entity" placeholders. You must treat them differently based on their context:
        * **CASE A: Intermediate Nodes (Bridges):** If a predicate implies a connection to another object (e.g., 'spouse', 'works_at', 'produced_by') and points to [Unnamed Entity], do **NOT** mark as FOUND. You must **Expand** this predicate. Unnamed Entities in this context indicate hidden nodes that may lead to the target, not targets themselves.
        * **CASE B: Attribute/End Nodes (Answers):** If a predicate represents the **specific attribute** requested (e.g., 'gender', 'birth_date', 'height') and points to [Unnamed Entity], **and is marked as an "end node"**, you **MUST mark this as FOUND**. The "Unnamed Entity" label simply means the value is currently hidden, but selecting the ID will retrieve the correct answer.

        **2. Handling Aggregations (Counts, Averages, Lists)**
        If the target requires a calculation (e.g., "How many children...", "Average age of..."), do **NOT** perform the calculation yourself.
        * **Action:** Select the predicates that provide the *raw data* for the calculation.
        * **Example:** If asked for "Number of employees", select all the 'employee' or 'works_at' predicates visible. Mark them as **FOUND**.

        ### Output Options

        **Option 1: Target Found**
        Select this if:
        * The text value explicitly answers the question.
        * The predicate matches the requested attribute (e.g., Target: "Gender", Predicate: 'gender'), even if the value is [Unnamed Entity] but is an end node.
        * The predicates provide the list of items needed to count or compute the answer.

        Format:
        Reasoning: [1-2 sentences explaining why this predicate contains the answer or raw data]
        Decision:
        {{FOUND}}
        {{ID}} {{ID}} ...

        **Option 2: Continue Searching (Expand)**
        Select this if the target is not currently visible, but expanding a relationship (like 'spouse', 'team', 'location') might lead to the answer.
        * Select up to 3 predicates.
        * Assign a confidence score (0.0 to 1.0).
        * Assign an [Alternative Path Score] to the branch representing the potential of the unselected predicates.

        Format:
        Reasoning: [Explain why expanding these paths will help reach the target]
        Decision: {{ID-Score}} {{ID-Score}} [BranchScore] [BranchScore]

        **Option 3: Give Up**
        If no predicates are relevant.
        Format: {{NONE}}

        ### Examples

        **Example 1: Explicit Value**
        Target: "Birth date of Albert Einstein"
        Predicates: 
        A1: 'birth_place' to [Ulm]
        A2: 'birth_date' to [1879-03-14]
        A3: 'date_of_birth' to [14 March 1879]
        Reasoning: Predicate A2 explicitly contains the date 1879-03-14. Predicate A3 also contains the same information in a different format. Both predicates directly answer the question.
        Decision:
        {{FOUND}}
        {{A2}} {{A3}}

        **Example 2: Hidden Attribute (The "Unnamed Entity" Exception)**
        Target: "Gender of the author of Harry Potter"
        Node: J.K. Rowling
        Predicates: 
        C1: 'notable_works' to [Harry Potter]
        C2: 'gender' to [Unnamed Entity] (This is an end node...)
        C3: 'residence' to [Scotland]
        Reasoning: The target asks for gender. Predicate C2 is the 'gender' attribute. Although the value is hidden as "Unnamed Entity", it is an end node that contains the answer.
        Decision:
        {{FOUND}}
        {{C2}}

        **Example 3: Need Expansion**
        Target: "The CEO of the company that created the iPhone"
        Node: iPhone
        Predicates:
        B1: 'designed_by' to [Apple Inc]
        B2: 'notable_people' to [Steve Jobs, Jony Ive, (and 15 more)]
        B3: 'cpu' to [A-Series]
        Reasoning: The node is iPhone. I need the CEO of the manufacturer. B1 leads to Apple Inc. I must expand B1 to find Apple's properties (like CEO). B2 also leads to relevant people, but it also includes many others, so I also consider it, but with lower confidence.
        Decision: {{B1-0.9}} {{B2-0.7}} [0.1]

        **Example 4: Aggregation (Count/List)**
        Target: "How many children does Angelina Jolie have?"
        Node: Angelina Jolie
        Predicates:
        D1: 'child' to [Maddox, Pax, Zahara (and more)]
        D2: 'occupation' to [Actress]
        Reasoning: I need to count the children. Predicates D1 provide the list of children needed to perform the count.
        Decision:
        {{FOUND}}
        {{D1}}
        
        **Example 5: Give Up**
        Target: "What is the population of Atlantis?"
        Node: Atlantis
        Predicates:
        E1: 'located_in' to [Ocean]
        E2: 'mythical_status' to [Mythical Place]
        Reasoning: None of the predicates provide information about population, nor can they lead to it.
        Decision:
        {{NONE}}
        
        Notes on formatting:
        - Decisions with multiple results are formatted with separate curly braces {{ }} for each part of the answer. You must follow this format exactly.
        - For selection you output IDs not labels or URIs.
""")
        return prompt
    
        #     **Example 6: Preference towards popular predicates**
        # Target: "Which music genres of music is Taylor Swift associated with?"
        # Predicates: 
        # A1: 'award_wins' to [11 Grammy Awards]
        # A2: 'music_genres' to [Pop, Country, Rock (and 1 more)]
        # A3: 'type_of_music' to [Pop]
        # Reasoning: Predicate A2 directly relates to music genres and contains multiple relevant values. Predicate A3 is also relevant but less commonly used. Therefore, I select both predicates to ensure comprehensive coverage of the target information.
        # Decision:
        # {{FOUND}}
        # {{A2}} {{A3}}
    
    def get_string_representation_of_frontier(self, state: SearchState) -> str:
        alphabetic_index = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        representation = ""
        for i, node in enumerate(state.frontier):
            if node.node_type == StepwiseSearchNodeType.LITERAL:
                continue
            representation += f"# Branch {alphabetic_index[i]}:\n"
            representation += f" Node: {node.label} ({node.value})\n"
            representation += f" Path to Node: {node.full_path_representation(readable=True) if node.previous else 'This is the root node.'}\n"
            representation += f" Predicates:\n"
            predicates = node.get_frontier_predicates()
            # if predicates is None or len(predicates) == 0:
            #     predicates = node.get_predicates()
            if len(predicates) == 0:
                representation += f"  (No predicates available for expansion from this node.)\n"
            #     raise ValueError(f"Predicate info not set for node {node.value}")
            for j, predicate in enumerate(predicates):
                representation += f"  {alphabetic_index[i]}{j}: '{predicate.get_label(kg=self.kg)}' {predicate.get_direction_word()} [{predicate.get_objects_string(readable=True, kg=self.kg)}]"
                objects_type = predicate.get_objects_sample_type()
                if objects_type == "Literal":
                    representation += f" (This is an end node, no further expansion possible.)"
                representation += "\n"
        return representation
    
    def generate_user_prompt(self, state: SearchState) -> str:
        frontier_str = self.get_string_representation_of_frontier(state)
        prompt = dedent(f"""\
Target Information: {state.question}
{frontier_str}

### Output
Use the specified output formats exactly:
- Option1: To denote a successful search write `{{FOUND}}` followed by `{{ID}}` on a new line. Not a label or URI, just the ID (e.g., `{{A0}}`).
- Option2: To continue the search `{{ID-Score}}` for expanding predicates and `[...]` for alternative confidence scores. A maximum of 3 predicates can be selected.
- Option3: To cancel the search because no paths are fruitful output `{{NONE}}`.

### Remember
- Be concise and focused in your reasoning.
- If a predicate directly contains the target information, use the FOUND option, even if other predicates seem relevant. You can select multiple predicates if they all contain the answer or different parts of it.
- If you choose to continue searching, select up to 3 predicates with confidence scores. Prioritize diversity and relevance. Continuing the search is preferred over giving up, but is not preferred over finding the answer.
- If no predicates are relevant and you are confident that no expansion would help, use the NONE option. If you want to backtrack, use the NONE option to cancel this search.
- Nodes with Unnamed Entities should only be marked as FOUND if they are end nodes containing the specific attribute requested. Otherwise, they should be expanded if relevant.""")
        print(prompt)
        return prompt
    
    
if __name__ == "__main__":
    import os
    from src.engine.config import CONFIG
    from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
    from src.elelem.provider import ProviderFactory
    from src.logging import create_console_logger, LogLevel
    
    create_console_logger(log_level=LogLevel.INFO)
    
    kg = KnowledgeGraphs.FREEBASE
    print("Loading KG...")
    kg.load(os.path.join(CONFIG().get("index_dir"), "freebase"))
    print("KG loaded.")
    
    llm = ProviderFactory.create_from_config(CONFIG())
    
    stepwise_search_manager = StepwiseSearchManager(kg, llm)
    
    # start_uri = "http://www.wikidata.org/entity/Q134798"
    start_uri = "http://rdf.freebase.com/ns/m.06c97"
    
    # predicates = stepwise_search_manager._get_predicate_info_for_node(start_uri, filter_literals=False)
    # for path in predicates:
    #     print(path.get_description(readable=True, kg=kg.value))
    
    # question = "In which languages were the books that Haruki Murakami authored?"
    # question = "What is the population of the country where Haruki Murakami was born?"
    question = "who was the brother of the wife of richard nixon?"
    
    # state = SearchState(question, start_uri, kg.get_kg_component(start_uri).label)
    # stepwise_search_manager.find_frontier_predicates(state)
    # representation = stepwise_search_manager.get_string_representation_of_frontier(state)
    # print(representation)
    
    # result = stepwise_search_manager.step(state)
    # print(result)
    
    # result = stepwise_search_manager.step(state)
    # print(result)
    
    # result = stepwise_search_manager.step(state)
    # print(result)
    
    # result = stepwise_search_manager.step(state)
    # print(result)
    
    result, paths, metrics = stepwise_search_manager.search(question, start_uri, max_steps=5)
    print("Search Result:", result)
    if result == StepwiseSearchResultEnum.FOUND:
        idx = 0
        for path in paths:
            triples_str = "\n".join(path.get_triples(readable=False))
            print(f"Path {idx}:\n{triples_str}")
            idx += 1
    
    # print(stepwise_search_manager.prepare_messages_for_llm(state))
    # print(stepwise_search_manager._user_prompt(state))
    
    # root = StepwiseSearchNode("http://example.org/entity/1", "Example Entity 1", None)
    # child = StepwiseSearchNode(["http://example.org/entity/2", "http://example.org/entity/3"], 
    #                    ["Example Entity 2", "Example Entity 3"], 
    #                    SearchConnection(root, 
    #                                     PredicateInfo("http://example.org/predicate/relatedTo", Direction.OUTGOING, None), 
    #                                     "related to", 
    #                                     None)
    #                    )
    # child2 = StepwiseSearchNode("Literal Value", "Literal Value", 
    #                     SearchConnection(child, 
    #                                      PredicateInfo("http://example.org/predicate/hasValue", Direction.OUTGOING, None), 
    #                                      "has value", 
    #                                      None)
    #                     )
    # print(child2.full_path_representation(readable=False))
    # print(child2.full_path_representation(readable=True))
    # print(child2.full_path_representation(readable=True, verbal=True))
    # print(child2.predicate_path_representation(readable=False))
    # print(child2.predicate_path_representation(readable=True))
    
    # print(child2.predicate_path_representation(readable=True, include_start=False, include_end=False))