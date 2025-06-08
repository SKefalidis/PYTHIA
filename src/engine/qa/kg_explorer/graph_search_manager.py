import textwrap
import time
from unittest import result
import requests
import regex as re

from typing import List, Any, Tuple
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

from src.elelem.provider import Provider, ProviderFactory
from src.engine.config import CONFIG
from src.metrics import get_kgaqa_tracker
from src.utils import execute_sparql_query, llm_call, embed
from src.logging import create_console_logger, log, LogComponent, LogLevel, LogType
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.knowledge_graphs.knowledge_graph import Direction, KgComponentType
from src.engine.qa.kg_explorer.graph_search_path import GraphSearchPath, GraphSearchPathPart, TargetType


    
class GraphSearchMetrics:
    def __init__(self):
        self.time = 0.0
        self.sparql_calls = 0
        self.sparql_time = 0.0
        self.llm_calls = 0
        self.llm_time = 0.0
        self.llm_inputs = 0
        self.llm_outputs = 0
        self.select = 0
        self.no_select = 0
        self.no_no = 0


class GraphSearchManager:
    
    class RepresentationType:
        ADHOC = "adhoc"
        LLM = "llm"
        TRIPLES = "triples"
        MULTILINE_TRIPLES = "multiline_triples"
        VERBALIZATION = "verbalization"
    
    class DualSearchHandle:
        """
        A helper object to manage the two results.
        """
        def __init__(self, winner_type: str, result_data: List[GraphSearchPath], metrics: GraphSearchMetrics, secondary_future=None):
            self.winner = winner_type  # 'shortest' or 'all'
            self.second = "shortest" if winner_type == "all" else "all"
            self.result: List[GraphSearchPath] = result_data  # The data available immediately
            self.metrics: GraphSearchMetrics = metrics
            self._secondary_future = secondary_future
            self._has_secondary_been_accessed = False

        def get_secondary_result(self) -> None | tuple[list[GraphSearchPath], GraphSearchMetrics]:
            """
            Returns the result of the second task.
            Blocks execution if the task is not finished yet.
            Returns None if the second task was cancelled.
            """
            if self._secondary_future is None:
                return None
            if self._has_secondary_been_accessed:
                return None
            if self._secondary_future:
                secondary_result, secondary_metrics = self._secondary_future.result()
                clean_secondary_result = [result for result in secondary_result if result not in self.result]
                self._has_secondary_been_accessed = True
                return clean_secondary_result, secondary_metrics
            return None
                
    
    def __init__(self, kg: KnowledgeGraphs, llm: str):
        self.kg: KnowledgeGraphs = kg
        self.llm = llm
        self.graph_search_server = CONFIG().get("endpoint_graph_search_server")
        
    def graph_search(self, sentence: str, start_uri: str, end_uri: str, representation: RepresentationType = RepresentationType.LLM, 
                     prune: bool = True, explain: bool = False, focus_on_recall: bool = False) -> Tuple[List[GraphSearchPath], GraphSearchMetrics]:
        metrics = GraphSearchMetrics()
        
        if self.kg.get_kg_component(start_uri) is None:
            log(f"Start URI not found in KG: {start_uri}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            return [], metrics
        if self.kg.get_kg_component(end_uri) is None:
            log(f"End URI not found in KG: {end_uri}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            return [], metrics
        
        if self.kg.value.is_class(start_uri) and self.kg.value.is_class(end_uri):
            log(f"Skipping graph search for class to class: {start_uri} to {end_uri}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            return [], metrics
        
        sentence = self.kg.get_label_from_uri(start_uri) + " " + sentence + " " + self.kg.get_label_from_uri(end_uri)
        
        graph_search_start_time = time.time()
        first_search_time = 0
        second_search_time = 0
        
        get_kgaqa_tracker()._pe_graph_search_calls += 1
        
        log(f"\t\t\tgraph search from {start_uri} to {end_uri}", LogComponent.PATH_EXTRACTOR, LogLevel.APPLICATION, LogType.NORMAL)
                
        dual_handle = self.parallel_graph_search(start_uri, end_uri)
        paths = dual_handle.result
        type = dual_handle.winner
        first_search_time = dual_handle.metrics.time
        metrics.sparql_calls += dual_handle.metrics.sparql_calls
        metrics.sparql_time += dual_handle.metrics.sparql_time
        
        if paths == []:
            result = dual_handle.get_secondary_result()
            if result is None:
                print("No secondary result available. Something went wrong...")
                return [], metrics
            paths, secondary_metrics = result
            type = dual_handle.second
            second_search_time = secondary_metrics.time
            metrics.sparql_calls += secondary_metrics.sparql_calls
            metrics.sparql_time += secondary_metrics.sparql_time
            
        # FILTER / NOT FILTER
        def prune_paths(paths: List[GraphSearchPath], type: str) -> List[GraphSearchPath]:
            ranked_paths = self._rank_paths_by_embedding_similarity(sentence, self.kg.value, paths)
            if type == 'all':
                paths = ranked_paths[:20]
            else:
                paths = ranked_paths[:10]
            return paths
        
        if prune:
            paths = prune_paths(paths, type)
        else:
            paths = paths
        
        def prepare_representations(paths: List[GraphSearchPath], representation: GraphSearchManager.RepresentationType) -> str:        
            if representation == self.RepresentationType.ADHOC:
                enumerated_candidates_string = "\n".join(f"{idx}. {path.get_path_description(readable=True, llm_friendly=False, show_length=False, show_specificity=True)}"
                                                        for idx, path in enumerate(paths))
            elif representation == self.RepresentationType.LLM:
                # for path in paths:
                    # path.find_sample_values(self.kg.value)
                enumerated_candidates_string = "\n".join(f"{idx}. {path.get_path_description(readable=True, llm_friendly=True, show_length=False, show_specificity=True, with_sample_values=False)}"
                                                        for idx, path in enumerate(paths))
            elif representation == self.RepresentationType.TRIPLES:
                enumerated_candidates_string = "\n".join(f"{idx}. {path.get_path_triples_description(readable=True, show_length=False, show_specificity=True)}"
                                                        for idx, path in enumerate(paths))
            elif representation == self.RepresentationType.VERBALIZATION:
                enumerated_candidates_string = "\n".join(f"{idx}. {path.verbalize_path(self.kg.value, alternative=False)}"
                                                        for idx, path in enumerate(paths))
            else:
                raise ValueError(f"Unknown representation type: {representation}")
            return enumerated_candidates_string
        
        paths_string = prepare_representations(paths, representation)
        
        # ask LLM for an assesment
        prompt = self._build_path_selection_prompt(sentence, start_uri, end_uri, paths_string, explain=explain, focus_on_recall=focus_on_recall)
        print(prompt)

        max_tokens = 300 if explain else 100
        
        llm_start = time.time()
        gen_tries = 3
        while True:
            response = llm_call(self.llm, prompt, max_tokens, temperature=0.0, return_usage=True)
            if response is not None:
                break
            else:
                gen_tries -= 1
                if gen_tries <= 0:
                    print("LLM call failed. Returning empty list.")
                    metrics.time += time.time() - graph_search_start_time + first_search_time + second_search_time
                    return [], metrics
        generated, usage = response
        print(f"Graph Search LLM output: {generated}")
        llm_end = time.time()
        
        metrics.llm_calls += 1
        metrics.llm_time += llm_end - llm_start
        metrics.llm_inputs += usage.prompt_tokens
        metrics.llm_outputs += usage.completion_tokens
        
        if r"{NO}" in generated:
            print("LLM indicated no good paths found.")
            try:
                new_paths, secondary_metrics = dual_handle.get_secondary_result()
            except:
                print("No secondary result available. Something went wrong...")
                return [], metrics
            type = dual_handle.second
            second_search_time = secondary_metrics.time
            metrics.sparql_calls += secondary_metrics.sparql_calls
            metrics.sparql_time += secondary_metrics.sparql_time
            
            # Remove already considered paths
            paths = [path for path in new_paths if path not in paths]
            
            if prune:
                paths = prune_paths(paths, type)
            else:
                paths = paths
                
            paths_string = prepare_representations(paths, representation)
            
            prompt = self._build_path_selection_prompt(sentence, start_uri, end_uri, paths_string, explain=explain, focus_on_recall=focus_on_recall)
            print(prompt)

            max_tokens = 300 if explain else 100
            
            llm_start = time.time()
            generated, usage = llm_call(self.llm, prompt, max_tokens, temperature=0.0, return_usage=True)
            print(f"Graph Search LLM output (secondary): {generated}")
            llm_end = time.time()
            
            metrics.llm_calls += 1
            metrics.llm_time += llm_end - llm_start
            metrics.llm_inputs += usage.prompt_tokens if usage and hasattr(usage, "prompt_tokens") else usage.input_tokens
            metrics.llm_outputs += usage.completion_tokens if usage and hasattr(usage, "completion_tokens") else usage.output_tokens
            
            if r"{NO}" in generated:
                metrics.no_no += 1
                print("LLM indicated no good paths found again. Returning empty list.")
                metrics.time += time.time() - graph_search_start_time + first_search_time + second_search_time
                return [], metrics
            else:
                metrics.no_select += 1
        else:
            metrics.select += 1
            endpoint = self.graph_search_server + "cancel"
            headers = {
                "Content-Type": "application/json"
            }
            response = requests.post(endpoint, headers=headers)
            if response.status_code == 200:
                log(f"Sent cancellation request for graph search tasks after first selection.", LogComponent.PATH_EXTRACTOR, LogLevel.APPLICATION, LogType.NORMAL)
            else:
                log(f"Failed to send cancellation request for graph search tasks. Status code: {response.status_code}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING, LogType.NORMAL)
                
        indices = re.findall(r"\{(\d+)\}", generated)
        selected_indices = set()
        for index in indices:
            idx = int(index)
            if 0 <= idx < len(paths):
                selected_indices.add(idx)
        paths = [paths[idx] for idx in selected_indices]
        
        metrics.time += time.time() - graph_search_start_time + first_search_time + second_search_time
        
        return paths, metrics

    def _build_path_selection_prompt(self, sentence: str, uri_start: str, uri_end: str, paths_string: str, explain: bool, focus_on_recall: bool) -> str:
        start_uri_label = self.kg.value.get_kg_component(uri_start).label
        end_uri_label = self.kg.value.get_kg_component(uri_end).label
        explain_text = "Briefly explain your decision." if explain else "Be concise and to the point."
        if focus_on_recall == False:
            answer_format = """Answer with the number of your chosen path inside curly brackets, e.g., {{2}}. 
If you have multiple good candidates but cannot decide the best answer with separate numbers, answer with all the numbers inside separate curly brackets, e.g., {{0}},{{3}},{{5}}.
You can select at most 3 paths. If none of the paths are good descriptors, answer with {{NO}}. 
If no path directly describes the relation in logical fashion and only implied connections are described, answer with {{NO}}."""
        else:
            answer_format = """You can select at most 3 paths. 
Unless there is a clear and unambiguous best path, prefer selecting multiple paths to maximize recall. 
At the same time do not select paths that are clearly not good descriptors of the relation. We care about recall but we don't discount precision entirely.
While selecting different paths try to select the most semantically appropriate paths but also try to have some diversity to maximize recall in the case where a path turns out to be not useful.
Answer with the index numbers of your selected path/paths inside separate curly brackets, e.g., {{0}},{{3}},{{5}}.
If none of the paths are good descriptors, answer with {{NO}}."""

        # Do try to select as few paths as possible, with the highest degree of confidence possible. Do not be afraid to not select any paths.
        prompt = textwrap.dedent(f"""### Task Description
You are given a question and a list of paths that connect two entities over a knowledge graph.
Your job is to evaluate which of the paths fully describes the relation between the two entities in the question.
The most important criteria is that the path describes the relation well and should be your main concern. The path should directly describe the relation, not imply it indirectly.
As additional information for each patch you are given:
- the number of matches for each path in the knowledge graph (popularity).
- the specificity score of each path (how specific the path is in the knowledge graph, higher is more specific).
These two properties are useful to assess the quality of each path. More matches means more support in the graph, but a low specificity might indicate that the path is too generic to be useful.
Among equally good candidates, prefer higher popularity if specificities are close (difference<1). If popularities are close, prefer higher specificity.

The question is: '{sentence}'
The two entities are: '{start_uri_label}' and '{end_uri_label}'
The paths are given as an indexed list below:
{paths_string}

### Answer Format
{answer_format}
{explain_text}

### Your Answer
""")
        return prompt
    
    def parallel_graph_search(self, uri_start: str, uri_end: str) -> DualSearchHandle:
        # 1. Start both functions in parallel threads
        # We use an executor to run them without blocking the main program yet
        executor = ThreadPoolExecutor(max_workers=2)
        
        future_shortest = executor.submit(self._shortest_paths, uri_start, uri_end)
        future_all = executor.submit(self._all_paths, uri_start, uri_end, filter_predicates=False, additional_hops=1)

        # 2. Block until the FIRST one finishes
        done, not_done = wait(
            [future_shortest, future_all], 
            return_when=FIRST_COMPLETED
        )

        # 3. Decision Logic
        if future_all in done:
            # CASE A: All Paths finished first
            # We explicitly discard shortest paths (conceptually cancelled)
            future_shortest.cancel() 
            result_all, metrics_all = future_all.result()
            return self.DualSearchHandle(
                winner_type='all', 
                result_data=result_all,
                metrics=metrics_all,
                secondary_future=None
            )

        elif future_shortest in done:
            # CASE B: Shortest Paths finished first
            # We return shortest paths immediately, but keep a handle on 'all_paths'
            result_shortest, metrics_shortest = future_shortest.result()
            return self.DualSearchHandle(
                winner_type='shortest', 
                result_data=result_shortest,
                metrics=metrics_shortest,
                secondary_future=future_all
            )
            
        # Fallback (should theoretically not happen if threads run correctly)
        return None
    
    @lru_cache(maxsize=16)
    def _all_paths(self, uri_start: str, uri_end: str, filter_predicates: bool, additional_hops: int) -> Tuple[List[GraphSearchPath], GraphSearchMetrics]:
        metrics = GraphSearchMetrics()
        
        if self.kg.value.is_class(uri_start) and self.kg.value.is_class(uri_end):
            log(f"Skipping all paths search for class to class: {uri_start} to {uri_end}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            return [], metrics
        
        if self.kg.value.is_class(uri_start) == None or self.kg.value.is_class(uri_end) == None:
            log(f"Could not determine if start or end node is a class: {uri_start}, {uri_end}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            return [], metrics
        
        get_kgaqa_tracker()._pe_all_paths_calls += 1
        start_time = time.time()

        endpoint = self.graph_search_server + "pythia-all-paths"

        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "uri_start": uri_start,
            "uri_end": uri_end,
            "fp": "1" if filter_predicates else "0",
            "additional_hops": additional_hops
        }

        try:
            metrics.sparql_calls += 1
            response = requests.post(endpoint, json=payload, headers=headers)
            metrics.sparql_time = time.time() - start_time
        except requests.exceptions.ConnectionError as e:
            print("\a")
            input(f"Connection error to graph search server at {endpoint}: {e}. You likely need to check if your endpoints are up. Press Enter to continue...")
            # start_time = time.time()
            # try:
            #     response = requests.post(endpoint, json=payload, headers=headers)
            #     metrics.sparql_time = time.time() - start_time
            # except requests.exceptions.ConnectionError as e:
            #     log(f"Error in graph search all paths request after retry: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            get_kgaqa_tracker()._pe_all_paths_time += time.time() - start_time
            metrics.sparql_time = time.time() - start_time
            metrics.time += time.time() - start_time
            return [], metrics
        except Exception as e:
            log(f"Error in graph search all paths request: {e} ({type(e).__name__})", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            get_kgaqa_tracker()._pe_all_paths_time += time.time() - start_time
            metrics.sparql_time = time.time() - start_time
            metrics.time += time.time() - start_time
            return [], metrics
        
        if response.status_code != 200:
            log(f"Graph search all paths request failed with status code {response.status_code}: {response.text}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            get_kgaqa_tracker()._pe_all_paths_time += time.time() - start_time
            metrics.time += time.time() - start_time
            return [], metrics
        
        data = response.json()
        if not 'result' in data:
            log(f"Graph search all paths request returned no results: {response.text}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            get_kgaqa_tracker()._pe_all_paths_time += time.time() - start_time
            metrics.time += time.time() - start_time
            return [], metrics
        data = data['result']
        if not 'predicates' in data or not 'directions' in data or not 'counts' in data:
            log(f"Graph search all paths request returned incomplete results: {response.text}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            get_kgaqa_tracker()._pe_all_paths_time += time.time() - start_time
            metrics.time += time.time() - start_time
            return [], metrics
        predicate_lists = data['predicates']
        predicate_directions = data['directions']
        path_matches = data['counts']
        
        candidate_paths = []
        for predicates, directions, count in zip(predicate_lists, predicate_directions, path_matches):
            path_parts = []
            for predicate, direction in zip(predicates, directions):
                direction_enum = Direction.OUTGOING if direction == "->" else Direction.INCOMING if direction == "<-" else Direction.UNKNOWN
                path_parts.append(GraphSearchPathPart(predicate, direction_enum))
            candidate_path = GraphSearchPath(kg=self.kg.value, parts=path_parts, popularity=count, leads_to=TargetType.URI, start_uri=uri_start, end_uri=uri_end)
            candidate_paths.append(candidate_path)
        
        get_kgaqa_tracker()._pe_all_paths_time += time.time() - start_time
        metrics.time += time.time() - start_time
        return candidate_paths, metrics

    @lru_cache(maxsize=16)
    def _shortest_paths(self, from_node: str, to_node: str, bidirectional: str = "true") -> Tuple[List[GraphSearchPath], GraphSearchMetrics]:
        metrics = GraphSearchMetrics()
        
        if self.kg.value.is_class(from_node) and self.kg.value.is_class(to_node):
            log(f"Skipping all paths search for class to class: {from_node} to {to_node}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            return [], metrics
        
        if self.kg.value.is_class(from_node) == None or self.kg.value.is_class(to_node) == None:
            log(f"Could not determine if start or end node is a class: {from_node}, {to_node}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            return [], metrics
        
        if from_node[0] == "<":
            from_node = from_node[1:-1]
        if to_node[0] == "<":
            to_node = to_node[1:-1]
        
        get_kgaqa_tracker()._pe_shortest_calls += 1
        start_time = time.time()
        
        from_node = self.kg.value.uril_to_uri(from_node)
        to_node = self.kg.value.uril_to_uri(to_node)
        query = f"""
        PREFIX path: <http://www.ontotext.com/path#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?predicatePath (COUNT(?predicatePath) AS ?pathCount)
        WHERE {{
          {{
            SELECT ?pathIndex (GROUP_CONCAT(STR(?p); SEPARATOR=" -> ") AS ?predicatePath)
            WHERE {{
              {{
                SELECT ?pathIndex ?p
                WHERE {{
                  VALUES (?src ?dst) {{
                    (<{from_node}> <{to_node}>)
                  }}
        
                  SERVICE path:search {{
                    [] path:findPath path:shortestPath ;
                       path:sourceNode ?src ;
                       path:destinationNode ?dst ;
                       path:pathIndex ?pathIndex ;
                       path:resultBindingIndex ?edgeIndex ;
                       path:resultBinding ?edge ;
                       path:propertyBinding ?p ;
                       path:poolSize 2;
                       path:bidirectional {bidirectional}.
                  }}
                }}
                ORDER BY ?pathIndex ?edgeIndex
              }}
            }}
            GROUP BY ?pathIndex
          }}
        }} GROUP BY ?predicatePath
        """
        # print(query)
        
        try:
            metrics.sparql_calls += 1
            query_result = execute_sparql_query(query, self.kg.endpoint)
            results = query_result.convert()
            metrics.sparql_time = time.time() - start_time
        except Exception as e:
            log(f"Error get_shortest_path_from_to (ran out of time, but this is important): {e}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            get_kgaqa_tracker()._pe_shortest_path_time += time.time() - start_time
            metrics.sparql_time = time.time() - start_time
            metrics.time += time.time() - start_time
            return [], metrics
        
        paths, popularity = [], []
        for result in results["results"]["bindings"]:
            paths.append((result["predicatePath"]["value"]))
            popularity.append(int(result["pathCount"]["value"]))
        log(f"Found {len(paths)} paths with get_shortest_path_from_to", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
        
        kg_paths = []
        for j in range(len(paths)):
            path_parts = []
            uris = paths[j].split(" -> ")
            for i in range(len(uris)):
                # print(f"before uril_to_uri: {uris[i]}")
                # uri = self.kg.value.uri_to_uril(uris[i])
                # print(f"after uril_to_uri: {uri}")
                path_parts.append(GraphSearchPathPart(uris[i], Direction.UNKNOWN))
            resolved_paths = self.resolve_directionally_ambiguous_path_parts(from_node, to_node, path_parts, popularity[j])
            kg_paths.extend(resolved_paths)

        get_kgaqa_tracker()._pe_shortest_path_time += time.time() - start_time
        
        metrics.time += time.time() - start_time
        
        return kg_paths, metrics
    
    def resolve_directionally_ambiguous_path_parts(self, start_uri: str, end_uri: str, ambiguous_parts: List[GraphSearchPathPart], pathCount: int) -> List[GraphSearchPath]:
        get_kgaqa_tracker()._pe_property_path_to_triples_calls += 1  
        log(f"Resolving ambiguous path from {start_uri} to {end_uri} with parts: {ambiguous_parts}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
        
        #
        # Generate triples from the path
        #
        triples = [""]
        unambiguous_path_parts: List[List[GraphSearchPathPart]] = [[]]
        current = "<" + start_uri + ">" # "<" + start + ">"
        var_index = 0
        for idx, ambiguous_part in enumerate(ambiguous_parts):
            new_triples = []
            new_unambiguous_path_parts: List[List[GraphSearchPathPart]] = []
            if idx == len(ambiguous_parts) - 1:
                new_var = "<" + end_uri + ">"  # "<" + goal + ">"
            else:
                new_var = "?var" + str(var_index)
                var_index += 1
            for t, parts in zip(triples, unambiguous_path_parts):
                if ambiguous_part.direction == Direction.UNKNOWN:
                    new_t_1 = t + current + " <" + ambiguous_part.predicate + "> " + new_var + " . \n"
                    new_triples.append(new_t_1)
                    new_unambiguous_path_parts.append(parts + [GraphSearchPathPart(ambiguous_part.predicate, Direction.OUTGOING)])
                    new_t_2 = t + new_var + " <" + ambiguous_part.predicate + "> " + current + " . \n"
                    new_triples.append(new_t_2)
                    new_unambiguous_path_parts.append(parts + [GraphSearchPathPart(ambiguous_part.predicate, Direction.INCOMING)])
                else:
                    raise Exception(f"Unknown direction: {parts[idx].direction}")
            current = new_var
            triples = new_triples
            unambiguous_path_parts = new_unambiguous_path_parts
        
        #
        # Only keep valid collections of triples, i.e., those that have results in the KG
        # TODO: Disabled because if the filter in the following query covers this too.
        # valid_triples = []
        # valid_unambiguous_path_parts = []
        # for triple, unambiguous_parts in zip(triples, unambiguous_path_parts):
        #     if self.kg.value.are_triples_valid(triple):
        #         valid_triples.append(triple)
        #         valid_unambiguous_path_parts.append(unambiguous_parts)
        
        #
        # Ignore triples that have a value connection
        #
        filtered_triples = []
        filtered_unambiguous_path_parts = []
        for triples_string, unambiguous_parts in zip(triples, unambiguous_path_parts):
            if self.kg.value.has_no_value_connection(triples_string) == False:
                continue
            filtered_triples.append(triples_string)
            filtered_unambiguous_path_parts.append(unambiguous_parts)
            # print(f"Keeping triples:\n{triples_string}")
            # print(f"With path parts: {unambiguous_parts}\n")
            
        kg_path_list = []
        for triples_string, unambiguous_parts in zip(filtered_triples, filtered_unambiguous_path_parts):
            kg_path = GraphSearchPath(kg=self.kg.value, parts=unambiguous_parts, popularity=pathCount, start_uri=start_uri, end_uri=end_uri) # The use of pathCount here is debatable. TODO
                                                                                                                  # Should we assign popularity differently when multiple paths are generated from one?
            kg_path_list.append(kg_path)
            
        log(f"Count of final kg paths: {len(kg_path_list)}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
            
        return kg_path_list
    
    def _rank_paths_by_embedding_similarity(self, question, kg, found_paths: List[GraphSearchPath]) -> List[GraphSearchPath]:
        if len(found_paths) == 0:
            return []
        
        question_embedding = embed(question, is_query=False)
        
        path_verbalizations = [path.verbalize_path(kg) for path in found_paths]
        path_embeddings = embed(path_verbalizations, is_query=False)
        
        path_similarities = cosine_similarity([question_embedding], path_embeddings)[0]
        ranked_paths = [path for _, path in sorted(zip(path_similarities, found_paths), key=lambda x: x[0], reverse=True)]
        
        return ranked_paths
    
    
if __name__ == "__main__":
    import os
    from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
    
    create_console_logger(log_level=LogLevel.INFO)
    
    kg = KnowledgeGraphs.WIKIDATA
    print("Loading KG...")
    kg.load(os.path.join(CONFIG().get("index_dir"), "wikidata"))
    print("KG loaded.")
    
    gsm = GraphSearchManager(kg, ProviderFactory.create_from_config(CONFIG()))
    
    # paths = gsm._all_paths("http://www.wikidata.org/entity/Q5", "http://www.wikidata.org/entity/Q571", False)
    # paths = gsm._all_paths("http://www.wikidata.org/entity/Q5", "http://www.wikidata.org/entity/Q15228", False)
    
    # start_time = time.time()
    paths, metrics = gsm._shortest_paths("http://www.wikidata.org/entity/Q37922", "http://www.wikidata.org/entity/Q134798")
    print(metrics.time)
    
    for idx, path in enumerate(paths):
        print(path.get_path_description(readable=True, llm_friendly=True, show_length=False, show_specificity=True))
        path.find_sample_values()
        print(path.get_path_description(readable=True, llm_friendly=True, show_length=False, show_specificity=True, with_sample_values=True))
    # end_time = time.time()
    # print(f"'shortest_paths' returned {len(paths)} paths in {end_time - start_time:.4f} seconds.")
    # for idx, path in enumerate(paths):
    #     print(f"{idx}. {path.get_path_description(readable=True, kg=kg, llm_friendly=True, show_length=False)}")
        
    # start_time = time.time()
    paths, metrics = gsm._shortest_paths("http://www.wikidata.org/entity/Q37922", "http://www.wikidata.org/entity/Q134798")
    print(metrics.time)
    # end_time = time.time()
    # print(f"'shortest_paths' returned {len(paths)} paths in {end_time - start_time:.4f} seconds.")
    # for idx, path in enumerate(paths):
    #     print(f"{idx}. {path.get_path_description(readable=True, kg=kg, llm_friendly=True, show_length=False)}")
        
    # start_time = time.time()
    # paths = gsm._all_paths("http://www.wikidata.org/entity/Q37922", "http://www.wikidata.org/entity/Q134798", filter_predicates=False, additional_hops=1)
    # end_time = time.time()
    # print(f"'all_paths' returned {len(paths)} paths in {end_time - start_time:.4f} seconds.")
    # for idx, path in enumerate(paths):
    #     print(f"{idx}. {path.get_path_description(readable=True, kg=kg, llm_friendly=True, show_length=False)}")
    
    # start_time = time.time()
    # dual_handle = gsm.parallel_graph_search("http://www.wikidata.org/entity/Q9826", "http://www.wikidata.org/entity/Q6711")
    # end_time = time.time()
    # print(f"'parallel_graph_search' returned {len(dual_handle.result)} paths in {end_time - start_time:.4f} seconds. Winner: {dual_handle.winner}")
    # dual_handle_secondary = dual_handle.get_secondary_result()
    # end_time2 = time.time()
    # print(f"'parallel_graph_search' secondary result returned {len(dual_handle_secondary)} paths in {end_time2 - end_time:.4f} seconds.")
    
    # selected = gsm.graph_search(
    #     sentence="Which High School did Allen Ginsberg attend?",
    #     start_uri="http://www.wikidata.org/entity/Q9826",
    #     end_uri="http://www.wikidata.org/entity/Q6711",
    #     representation=GraphSearchManager.RepresentationType.ADHOC
    # )
    
    # print(f"Selected {len(selected)} paths:")
    # for idx, path in enumerate(selected):
    #     print(f"{idx}. {path.get_path_description(readable=True, kg=kg, llm_friendly=False, show_length=False)}")
    
    # selected = gsm.graph_search(
    #     sentence="Which High School did Allen Ginsberg attend?",
    #     start_uri="http://www.wikidata.org/entity/Q9826",
    #     end_uri="http://www.wikidata.org/entity/Q6711",
    #     representation=GraphSearchManager.RepresentationType.LLM
    # )
    
    # print(f"Selected {len(selected)} paths:")
    # for idx, path in enumerate(selected):
    #     print(f"{idx}. {path.get_path_description(readable=True, kg=kg, llm_friendly=False, show_length=False)}")
    
    # selected = gsm.graph_search(
    #     sentence="Which High School did Allen Ginsberg attend?",
    #     start_uri="http://www.wikidata.org/entity/Q9826",
    #     end_uri="http://www.wikidata.org/entity/Q6711",
    #     representation=GraphSearchManager.RepresentationType.TRIPLES
    # )
    
    # print(f"Selected {len(selected)} paths:")
    # for idx, path in enumerate(selected):
    #     print(f"{idx}. {path.get_path_description(readable=True, kg=kg, llm_friendly=False, show_length=False)}")
    