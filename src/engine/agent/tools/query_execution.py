import json
from enum import Enum
from typing import Tuple, Any, List

from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.engine.agent.tools.tool import Tool
from src.engine.gost_requests import format_query, validate_query, validate_query_with_errors
from src.utils import execute_sparql_query, is_uri
from src.engine.gost_requests import remove_filters, remove_having, extract_triples


class QueryExecutionErrors(Enum):
    NO_ERROR = "No error."
    UNRESOLVED_PREFIX = "Unresolved prefix in SPARQL query."
    UNKNOWN_ERROR = "Unknown error."
    SERVICE_USAGE = "Usage of SERVICE in SPARQL query."

class EmptyQueryResultsReasons(Enum):
    DUE_TO_FILTERS = 1
    DUE_TO_GRAPH_PATTERN_INVALID_TRIPLES = 2
    DUE_TO_GRAPH_PATTERN_INVALID_COMBINATION = 3
    DUE_TO_SELECT_VARS = 4
    UNKNOWN = 5
    
class QueryExecutionTool(Tool):
    
    def __init__(self, kg: KnowledgeGraphs, enable_explanation_of_empty_results: bool = True):
        super().__init__()
        self.kg = kg
        self.prefixes = self.kg.value.prefixes
        self.enable_explanation_of_empty_results = enable_explanation_of_empty_results
        self.empty_due_to_filters_count = 0
        self.empty_due_to_invalid_triples_count = 0
        self.empty_due_to_invalid_combination_count = 0
        self.empty_due_to_select_vars_count = 0
        self.empty_unknown_count = 0
        print(f"Initialized QueryExecutionTool with enable_explanation_of_empty_results={enable_explanation_of_empty_results}")
    
    @classmethod
    def name(cls) -> str:
        return "execute_query"
    
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "execute_query",
                "description": "Execute a SPARQL query and retrieve a sample of its results if successful. If not, an error message is returned.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The SPARQL query to execute."},
                    },
                    "required": ["query"],
                },
            },
        }
    
    def function(self, *args, **kwargs) -> Tuple[bool, str, str]:
        query = kwargs.get("query")
        if query is None or query.strip() == "":
            return False, query, "No SPARQL query provided. Provide a valid SPARQL query via the 'query' parameter."
        
        error_type, validation_message = self._parse_query(query)
        
        if error_type == QueryExecutionErrors.NO_ERROR:
            try:
                results = self._get_query_results(query)
            except Exception as e:
                return False, query, f"Error executing SPARQL query: {str(e)}"
            return True, query, self.prepare_results_for_output(query=query, results=results)
        else:
            if error_type == QueryExecutionErrors.UNRESOLVED_PREFIX:
                query = self.prefixes + "\n" + query
                error_type, validation_message = self._parse_query(query) # ignore second validation message, if it failed we revert to the original
                if error_type == QueryExecutionErrors.NO_ERROR:
                    query = format_query(query)
                    results = self._get_query_results(query)
                    return True, query, self.prepare_results_for_output(query=query, results=results)
                else:
                    return False, query, validation_message
            else:
                return False, query, validation_message
            
    def prepare_results_for_output(self, query: str, results: Any) -> str:
        if len(results) == 0:
            reasons, explanation = self._explain_empty_results(query=query) if self.enable_explanation_of_empty_results else ("", "")
            if self.enable_explanation_of_empty_results:
                if reasons == EmptyQueryResultsReasons.DUE_TO_FILTERS:
                    self.empty_due_to_filters_count += 1
                elif reasons == EmptyQueryResultsReasons.DUE_TO_GRAPH_PATTERN_INVALID_TRIPLES:
                    self.empty_due_to_invalid_triples_count += 1
                elif reasons == EmptyQueryResultsReasons.DUE_TO_GRAPH_PATTERN_INVALID_COMBINATION:
                    self.empty_due_to_invalid_combination_count += 1
                elif reasons == EmptyQueryResultsReasons.DUE_TO_SELECT_VARS:
                    self.empty_due_to_select_vars_count += 1
            return "The query executed successfully but returned no results. " + explanation
        
        uris_per_result = [[] for _ in results]
        for i, row in enumerate(results):
            if isinstance(row, str):
                continue
            for key, value in row.items():
                if isinstance(value, str) and value.startswith("http"):
                    previous = value
                    row[key] = self.kg.value.shorten_uri(value)
                    if len(previous) > len(row[key]):
                        uris_per_result[i].append(previous)
        
        if len(results) > 10:
            shortened_results = results[:5] + results[-5:]
            message = f"The query executed successfully and returned {len(results)} results. Showing first 5 and last 5 results:\n"
            for r in shortened_results:
                message += json.dumps(r) + "\n"
            uris_per_result = uris_per_result[:5] + uris_per_result[-5:]
            if any(len(uris) > 0 for uris in uris_per_result):
                message += f"\nThe following URIs were found in the results:\n"
            for uris in uris_per_result:
                for uri in uris:
                    short_uri = self.kg.value.shorten_uri(uri)
                    label = self.kg.get_label_from_uri(uri)
                    message += f"- {short_uri} : {label}\n"
            message += f"Remember, if the query returns the correct results, you don't need to refine it to retrieve labels or additional information. You can use the returned URIs directly in your final answer."
            return message
        else:
            message = f"The query executed successfully and returned {len(results)} results:\n"
            for r in results:
                message += json.dumps(r) + "\n"
            if any(len(uris) > 0 for uris in uris_per_result):
                message += f"\nThe following URIs were found in the results:\n"
            for uris in uris_per_result:
                for uri in uris:
                    short_uri = self.kg.value.shorten_uri(uri)
                    label = self.kg.get_label_from_uri(uri)
                    message += f"- {short_uri} : {label}\n"
            message += f"Remember, if the query returns the correct results, you don't need to refine it to retrieve labels or additional information. You can use the returned URIs directly in your final answer."
            return message
    
    def _parse_query(self, query: str) -> Tuple[QueryExecutionErrors, str]:
        if "SERVICE wikibase" in query.upper():
            return QueryExecutionErrors.SERVICE_USAGE, "Usage of the SERVICE keyword in SPARQL query is not allowed."
        result = validate_query_with_errors(query)
        if result == None:
            return QueryExecutionErrors.UNKNOWN_ERROR, "An unknown error occurred during query validation."
        if "No errors were found" in result:
            return QueryExecutionErrors.NO_ERROR, "Valid SPARQL query."
        elif "Unresolved prefix" in result:
            return QueryExecutionErrors.UNRESOLVED_PREFIX, result
        else:
            return QueryExecutionErrors.UNKNOWN_ERROR, result  # General error for now
        
    def _get_query_results(self, query):
        """
        Executes a SPARQL query and returns a clean, token-efficient string 
        optimized for OpenAI tool calling. Handles SELECT, ASK, and CONSTRUCT.
        """
        # print(f"Endpoint:\n{self.kg.value.endpoint}")
        # print(f"Query:\n{query}")
        
        try:
            results = execute_sparql_query(query, self.kg.value.endpoint, max_wait_minutes=1)
            results = results.convert()
            # BRANCH A: ASK Queries (return boolean)
            # Standard SPARQL JSON for ASK looks like: { "head": {}, "boolean": true }
            if 'boolean' in results:
                return [str(results['boolean'])]  # Returns "True" or "False"

            # BRANCH B: SELECT Queries (return bindings)
            # Standard SPARQL JSON for SELECT looks like: { "results": { "bindings": [...] } }
            if 'results' in results:
                bindings = results['results']['bindings']
                simplified_list = []
                
                for row in bindings:
                    clean_item = {}
                    for key, value_obj in row.items():
                        # We strip the metadata (type, datatype) and just keep the value
                        clean_item[key] = value_obj['value']
                    if len(clean_item) > 0:
                        simplified_list.append(clean_item)
                    
                # TODO: If no results, show why (e.g., no matching data, filtered out, etc.)

                return simplified_list

        except Exception as e:
            print(f"Error processing SPARQL results: {str(e)}")
            return []
        
    def _explain_empty_results(self, query: str) -> Tuple[EmptyQueryResultsReasons, str]:
        # check if removing filters or having clauses brings results. This means the query is correct if the filters are also correct.
        query_without_filtering = remove_having(remove_filters(query))
        filtered_results = self._get_query_results(query_without_filtering)
        if len(filtered_results) > 0:
            return EmptyQueryResultsReasons.DUE_TO_FILTERS, "All results are filtered out by FILTER or HAVING clauses. If the filters are correct then the query is valid. Otherwise, check the filter conditions."
        
        # check if the triples have any results at all. If not, break down the triples and check them one by one.
        try:
            raw_triples = extract_triples(query)
            # print(f"Extracted triples:\n{raw_triples}")
            triples_tuples = []
            for triple_line in raw_triples.split('\n'):
                triple_parts = triple_line.split('\t\t')
                subject = triple_parts[0]
                # print(f"Triple parts: {triple_parts}")
                if subject.startswith("?"):
                    subject = subject
                else:
                    new_subj = self.kg.value.shorten_uri(subject)
                    if new_subj != subject:
                        subject = new_subj
                    else:
                        if is_uri(subject) and subject.startswith("http://"):
                            subject = '<' + subject + '>'
                        else:
                            subject = subject  # leave as is, might be a literal
                predicate = triple_parts[1]
                if predicate.startswith("?"):
                    predicate = predicate
                else:
                    new_pred = self.kg.value.shorten_uri(predicate)
                    if new_pred != predicate:
                        predicate = new_pred
                    else:
                        if is_uri(predicate) and predicate.startswith("http://"):
                            predicate = '<' + predicate + '>'
                        else:
                            predicate = predicate  # leave as is, might be a literal
                obj = triple_parts[2]
                if obj.startswith("?"):
                    obj = obj
                else:
                    new_obj = self.kg.value.shorten_uri(obj)
                    if new_obj != obj:
                        obj = new_obj
                    else:
                        if is_uri(obj) and obj.startswith("http://"):
                            obj = '<' + obj + '>'
                        else:
                            obj = obj  # leave as is, might be a literal
                triples_tuples.append( (subject, predicate, obj) )
            triples_string = ""
            for s, p, o in triples_tuples:
                triples_string += f"  {s} {p} {o} .\n"
                
            print(f"Triples to check for results: {triples_string}")
            if not validate_query(self.kg.prefixes + "SELECT * WHERE {\n" + triples_string + "\n}"):
                self.empty_unknown_count += 1
                return EmptyQueryResultsReasons.UNKNOWN, "Could not identify the reason for empty results."
                
            triples_results = self.kg.value.get_values_for_triples(triples_string, k=5, prefixes=self.prefixes)
            if len(triples_results) == 0:
                # if the triples themselves return no results, then it is likely that the combination of triples is incorrect.
                invalid_singular_triples: List[str] = []
                valid_inverted_triples: List[str] = []
                for s, p, o in triples_tuples:
                    single_triple_string = f"  {s} {p} {o} .\n"
                    valid = self.kg.value.are_triples_valid(single_triple_string, prefixes=self.prefixes)
                    if valid:
                        continue
                    invalid_singular_triples.append( (s, p, o) )
                    # check inverse
                    inverted_triple_string = f"  {o} {p} {s} .\n"
                    valid_inverted = self.kg.value.are_triples_valid(inverted_triple_string, prefixes=self.prefixes)
                    if valid_inverted:
                        valid_inverted_triples.append( (o, p, s) )
                    else:
                        valid_inverted_triples.append( None )

                if len(invalid_singular_triples) > 0:
                    msg = "The query returns no results due to invalid triples in the graph pattern:"
                    for i, (s, p, o) in enumerate(invalid_singular_triples):
                        msg += f"\n- Triple: {s} {p} {o}"
                        if valid_inverted_triples[i] is not None:
                            inv_s, inv_p, inv_o = valid_inverted_triples[i]
                            msg += f" (Note: the inverted triple {inv_s} {inv_p} {inv_o} is valid.)"
                    return EmptyQueryResultsReasons.DUE_TO_GRAPH_PATTERN_INVALID_TRIPLES, msg
                
                # check combinations of triples
                combined_triples = ""
                last_valid_triples = ""
                for s, p, o in triples_tuples:
                    combined_triples += f"  {s} {p} {o} .\n"
                    valid = self.kg.value.are_triples_valid(combined_triples, prefixes=self.prefixes)
                    if valid:
                        last_valid_triples = combined_triples
                    else:
                        break
                if last_valid_triples.strip() != "":
                    msg = "This is caused by an invalid combination of triples in the graph pattern (the individual triples are valid). The following subset of triples is valid:\n"
                    msg += last_valid_triples

                return EmptyQueryResultsReasons.DUE_TO_GRAPH_PATTERN_INVALID_COMBINATION, msg
            
            # if the triples themselves return results, then it is likely that the select variables are incorrect.
            return EmptyQueryResultsReasons.DUE_TO_SELECT_VARS, "This is likely due to incorrect SELECT variables. Check that the variables in the SELECT clause match those used in the WHERE clause."
        except Exception as e:
            print(f"Error explaining empty results: {str(e)}")
            self.empty_unknown_count += 1
            return EmptyQueryResultsReasons.UNKNOWN, "Could not identify the reason for empty results."