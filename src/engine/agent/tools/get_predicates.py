from functools import lru_cache
import json
from typing import Tuple, Any, List

from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.engine.agent.tools.tool import Tool
from src.knowledge_graphs.knowledge_graph import KgClass, KgComponent, KgEntity, PredicateInfo, Direction
from src.utils import embed, cosine_similarity, is_uri, execute_sparql_query


class PredicatesTool(Tool):
    
    def __init__(self, kg: KnowledgeGraphs):
        super().__init__()
        self.kg = kg
    
    @classmethod
    def name(cls) -> str:
        return "get_predicates_for_node"
    
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "get_predicates_for_node",
                "description": "Returns the 10 most similar predicates of a node (URI) for the given label. Useful to explore the properties and relations of an entity or class in the knowledge graph. It uses semantic similarity to rank the predicates based on a provided label.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uri": {"type": "string", "description": "The URI of the node to get predicates for."},
                        "label": {"type": "string", "description": "The predicate description to use for semantic similarity."},
                    },
                    "required": ["uri", "label"],
                },
            },
        }
    
    def function(self, *args, **kwargs) -> str:
        uri = kwargs.get("uri")
        if uri is None or uri.strip() == "":
            return "No URI provided. Provide a valid URI via the 'uri' parameter."
        label = kwargs.get("label")
        if label is None or label.strip() == "":
            return "No label provided. Provide a valid label via the 'label' parameter."
        
        if not is_uri(uri):
            expanded = self.kg.value.expand_uri(uri)
            if not is_uri(expanded):
                return "Incorrect usage of stepwise_search: start URI is invalid. Make sure to provide a full URI, without angle brackets."
            else:
                uri = expanded
        
        kgc = self.kg.value.get_kg_component(uri)
        predicates = []
        entity_predicates = []
        if kgc is None:
            predicates: List[PredicateInfo] = KgEntity.get_predicates_for_entity(uri, self.kg.endpoint, filter_literals=False, limit=1_000_000) 
        else:
            if kgc.is_entity():
                kgc: KgEntity
                print("Retrieving predicates for entity node:", uri)
                predicates: List[PredicateInfo] = kgc.get_predicates(self.kg.endpoint, filter_literals=False, limit=1_000_000)
            elif kgc.is_class():
                kgc: KgClass
                print("Retrieving predicates for class node:", uri)
                predicates: List[PredicateInfo] = kgc.get_own_predicates()
                for predicate in predicates:
                    objects = self.kg.value.get_object_for_node_predicate_info(uri, predicate)
                    predicate.objects = objects
                entity_predicates: List[PredicateInfo] = kgc.get_entity_predicates()
            else:
                predicates = []
        if len(predicates) == 0 and entity_predicates == 0:
            return "No predicates found for the given URI."
        
        ranked_predicates = self.rank_predicates(predicates, label)
        top_k = min(10, len(ranked_predicates))
        top_predicates = ranked_predicates[:top_k]
        
        
        ranked_entity_predicates = self.rank_predicates(entity_predicates, label)
        top_k = min(10, len(ranked_entity_predicates))
        top_entity_predicates = ranked_entity_predicates[:top_k]
        
        message = self.prepare_results_for_output(uri, top_predicates, top_entity_predicates)
        
        return message
    
    def rank_predicates(self, predicates: List[PredicateInfo], label: str) -> List[PredicateInfo]:
        if len(predicates) == 0:
            return []
        
        question_embedding = embed(label, is_query=False)
        
        predicate_verbalizations = [predicate.get_label(self.kg.value) for predicate in predicates]
        predicate_embeddings = embed(predicate_verbalizations, is_query=False)
        
        predicate_similarities = cosine_similarity([question_embedding], predicate_embeddings)[0]
        ranked_predicates = [predicate for _, predicate in sorted(zip(predicate_similarities, predicates), key=lambda x: x[0], reverse=True)]
        
        return ranked_predicates
            
    def prepare_results_for_output(self, uri, results: List[PredicateInfo], entity_results: List[PredicateInfo]) -> str:                
        message = f"Predicates discovered for {self.kg.value.shorten_uri(uri)}:\n"
        og_uri = uri
        uri = self.kg.value.shorten_uri(uri)
        for r in results:
            if r.direction == Direction.OUTGOING:
                message += f"- {uri} {self.kg.value.shorten_uri(r.uri)} ({r.get_label(self.kg.value)}) ?x\n"
                message += f"  x: [{r.get_objects_string(True, self.kg.value, with_uri=True)}]\n"
            else:
                message += f"- ?x {self.kg.value.shorten_uri(r.uri)} ({r.get_label(self.kg.value)}) -> {uri}\n"
                message += f"  x: [{r.get_objects_string(True, self.kg.value, with_uri=True)}]\n"
        if len(entity_results) > 0:
            type_predicate = "rdf:type" if self.kg != KnowledgeGraphs.WIKIDATA else "wdt:P31"
            kgc: KgClass = self.kg.value.get_kg_component(og_uri)
            message += f"\n Because {self.kg.value.shorten_uri(uri)} is a class we also retrieve predicates for its instances ({kgc.incoming_edges_count} via {type_predicate}):\n"
            for r in entity_results:
                instance_values = self.get_samples_for_class_instance_predicate(og_uri, r.uri, r.direction, sample_size=3)
                if r.direction == Direction.OUTGOING:
                    message += f"- ?inst {self.kg.value.shorten_uri(r.uri)} ({r.get_label(self.kg.value)}) ?x [matches: {r.cardinality}]\n"
                else:
                    message += f"- ?x {self.kg.value.shorten_uri(r.uri)} ({r.get_label(self.kg.value)}) ?inst [matches: {r.cardinality}]\n"
                message += f"   Samples:\n"
                for inst, val in instance_values:
                    instance_uri = self.kg.value.shorten_uri(inst)
                    instance_label =  self.kg.get_label_from_uri(inst)
                    value = val
                    if is_uri(val):
                        value = self.kg.value.shorten_uri(val)
                        value_label = self.kg.get_label_from_uri(val)
                        message += f"    {{inst: {instance_uri} ({instance_label}), value: {value} ({value_label})}}\n"
                    else:
                        message += f"    {{inst: {instance_uri} ({instance_label}), value: {value}}}\n"
        return message
    
    @lru_cache(maxsize=1024)
    def get_samples_for_class_instance_predicate(self, class_uri: str, predicate_uri: str, direction: Direction, sample_size: int = 3) -> List[Tuple[str, Any]]:
        type_predicate = "rdf:type" if self.kg != KnowledgeGraphs.WIKIDATA else "wdt:P31"
        if direction == Direction.OUTGOING:
            query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT ?instance ?value WHERE {{
                ?instance {type_predicate} <{class_uri}> .
                ?instance <{predicate_uri}> ?value .
            }} LIMIT {sample_size}
            """
        else:
            query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT ?instance ?value WHERE {{
                ?instance <{type_predicate}> <{class_uri}> .
                ?value <{predicate_uri}> ?instance .
            }} LIMIT {sample_size}
            """
        query_result = execute_sparql_query(query, self.kg.endpoint)
        results = query_result.convert()
        instance_values = []
        for result in results["results"]["bindings"]:
            instance = result["instance"]["value"]
            value = result["value"]["value"]
            instance_values.append((instance, value))
        return instance_values
        
if __name__ == "__main__":
    
    from src.engine.config import CONFIG
    from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
    from src.logging import create_console_logger
    import os
    
    create_console_logger()

    kg = KnowledgeGraphs.FREEBASE
    kg.load(os.path.join(CONFIG().get("index_dir"), "freebase"))

    tool = PredicatesTool(kg)

    result = tool.function(
        uri="http://rdf.freebase.com/ns/m.01mp",  
        label="borders"
    )
    print(result)