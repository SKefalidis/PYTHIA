# Using LLMs to generate SPARQL queries for the given question, without context.
import textwrap
import re
from typing import List, Tuple
from time import time
import time as _sleep_time

from litellm import completion
from src.knowledge_graphs.knowledge_graph import KnowledgeGraph
from src.engine.qa.query_generator.query_db import QueryDb
from src.engine.gost_requests import expand_query_prefixes
from src.metrics import PerformanceMetrics

class LlmSparqlBaseline:
    
    def __init__(self, model: str, kg_name: str, kg: KnowledgeGraph, use_cot: bool, use_few_shot: bool, query_db: QueryDb):
        self.model = model
        self.kg_name = kg_name
        self.kg = kg
        self.use_cot = use_cot
        self.use_few_shot = use_few_shot
        self.query_db = query_db

    def generate_sparql(self, question, **kwargs) -> Tuple[str, PerformanceMetrics]:
        """
        Generates a SPARQL query for the given question using the LLM.
        
        Returns: (generated_sparql, usage).
        """
        messages = self.create_prompt(question, **kwargs)
        start_time = time()

        max_retries = 10
        wait_seconds = 10
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                max_tokens = 500
                reasoning_effort = None
                if "gpt-5" in self.model:
                    max_tokens = 1000
                    reasoning_effort = "low"
                elif "gpt-oss" in self.model:
                    max_tokens = 1500
                    reasoning_effort = "medium"
                response = completion(model=self.model,
                                      reasoning_effort=reasoning_effort,
                                      messages=messages,
                                      max_tokens=max_tokens,)
                break
            except Exception as e:
                print(e)
                print(f"Attempt {attempt} failed. Retrying in {wait_seconds} seconds...")
                if attempt == max_retries:
                    raise
                _sleep_time.sleep(wait_seconds)
                
        messages.append(response['choices'][0]['message']['content'])
        end_time = time()
        sparql_query = self.extract_sparql_from_response(response['choices'][0]['message']['content'])

        def replace_urils_with_uris(query: str) -> str|None:
            query = expand_query_prefixes(query)
            if query is None:
                return None
            return self.kg.triples_with_urils_to_triples_with_uris(query)
        
        query = replace_urils_with_uris(sparql_query)
        if query is None:
            query = replace_urils_with_uris(self._basic_prefixes() + sparql_query)
        if query is None:
            print("Warning: Could not replace URILs with URIs in the generated query.")
            print("Response:")
            print(response['choices'][0]['message']['content'])
        sparql_query = query if query is not None else sparql_query
        
        usage = response['usage']
        
        return sparql_query, messages, PerformanceMetrics(0, 0, 1, end_time - start_time, usage['prompt_tokens'], usage['completion_tokens'])

    def extract_sparql_from_response(self, response: str) -> str:
        """Extracts the SPARQL query from the LLM response."""
        pattern = r"```(sparql)?\s*(.*?)\s*```"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(2).strip()
        else:
            # If no code block is found, assume the entire response is the query
            return response.strip()
    
    def _system_prompt(self):
        return textwrap.dedent(
            f"""
                You are an expert SPARQL user tasked with generating SPARQL queries that accurately answer the given natural language questions over the knowledge graph {self.kg_name}.
                - Ensure that the SPARQL queries that you write are syntactically correct and make use of appropriate entities, classes, and predicates.
                - Use the given classes, entities and predicates either as full URIs with angle brackets (<>) or as prefixed names (you must make sure that the PREFIX is defined).
                - Be careful with commas and other special characters in URIs. Those can't be prefixed (for example, do not write ex:New_York,_U.S. instead use <http://example.org/New_York,_U.S.>).
                - Make careful use of FILTERs, OPTIONALs, and other SPARQL constructs as needed to accurately capture the intent of the question.
                - Answer YES/NO questions with ASK queries, and other types of questions with SELECT queries.
                - Return only the SPARQL query without any additional explanations or text, surrounded by triple backticks (```).
                - Do not use any non-standard SPARQL extensions or SERVICEs; stick to standard SPARQL 1.1 syntax on a vanilla RDF store endpoint. Do not use WIKIBASE services.
            """)
                
    def _few_shot_system_prompt(self):
        return "To help you generate accurate SPARQL queries, the user will give you three examples of questions and their corresponding SPARQL queries.\
                If the question is similar to one of the examples, try to follow the same structure and approach in your generated query."
    
    def _cot_system_prompt(self):
        return "**Think step by step and explain your reasoning before providing the final SPARQL query. Only the final query must be contained in triple backticks (```).**"
    
    def _user_prompt(self, question: str, **kwargs):
        return f"Generate a SPARQL query for the following question:\nQuestion: {question}\nSPARQL Query:"
    
    def _few_shot_user_prompt(self, question: str, examples: List[Tuple[str, str]], **kwargs):
        examples_string = "\n" + '\n'.join([f"Question: {ex[0]}\nSPARQL Query:\n```{ex[1]}```" for ex in examples]) + "\n"
        user_prompt = self._user_prompt(question, **kwargs)
        return textwrap.dedent(
            f"""I want you to generate a SPARQL query for the following question: {question}
                Here are some examples to guide you:
                {examples_string}
                {user_prompt}""")
    
    def create_prompt(self, question: str, **kwargs) -> List[dict]:
        """Constructs LLM prompt programmatically depending on class parameters."""
        if self.use_few_shot:
            examples = self.query_db.get_relevant_queries(question)
        
        system_prompt_str = self._system_prompt()
        if self.use_few_shot:
            system_prompt_str += "\n" + self._few_shot_system_prompt()
        if self.use_cot:
            system_prompt_str += "\n" + self._cot_system_prompt()

        messages = []
        messages.append({"role": "system", "content": system_prompt_str})

        if self.use_few_shot:
            messages.append({"role": "user", "content": self._few_shot_user_prompt(question, examples, **kwargs)})
        else:
            messages.append({"role": "user", "content": self._user_prompt(question, **kwargs)})
        
        return messages
    
    def _basic_prefixes(self) -> str:
        return textwrap.dedent(
            """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            """)