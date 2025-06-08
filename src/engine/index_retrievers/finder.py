from functools import lru_cache
from typing import List
import os
import re

from src.knowledge_graphs.knowledge_graph import Direction, KgClass, KgComponentType, PredicateInfo, StartPoint
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.engine.index_retrievers.index_retrievers import FaissIndexRetriever, HybridIndexRetriever, IndexRetriever, Similarity, SimstringIndexRetriever
from src.utils import execute_sparql_query, embed, get_embed_model, get_kgaqa_tracker
from src.logging import LogLevel, LoggingOptions, create_logger, log
from sentence_transformers.util import cos_sim
from time import time

SPARQL_MATCH_ENTITIES_WITH_PREDICATES = """
SELECT DISTINCT ?entity ?predicate WHERE {{
    VALUES ?entity {{ {entities} }}
    VALUES ?predicate {{ {predicates} }}
    ?entity ?predicate ?object .
}}"""

SPARQL_MATCH_CLASSES_WITH_PREDICATES = """
SELECT DISTINCT ?class ?predicate WHERE {{
    VALUES ?class {{ {classes} }}
    VALUES ?predicate {{ {predicates} }}
    ?instance a ?class .
    ?instance ?predicate ?object .
}}"""

SPARQL_MATCH_CLASSES_WITH_PREDICATES_WIKIDATA = """
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT DISTINCT ?class ?predicate WHERE {{
    VALUES ?class {{ {classes} }}
    VALUES ?predicate {{ {predicates} }}
    ?instance wdt:P31 ?class .
    ?instance ?predicate ?object .
}}"""

class Finder:
    """
    Intended uses:
    1. Unified query: single query for classes, entities, properties. The Agent can only issue one query at a time, and its results are combined.
    2. Separate queries: different query for classes, entities, properties. The Agent can issue separate queries for each type, and their results are combined.
    *. Multiple queries: The agent can pass query lists, and all results are combined (modification of above two).
    """
    
    def __init__(self, kg: KnowledgeGraphs, k: int = None):
        self.kg = kg
        self.k = k
        self.class_simstring_retriever: SimstringIndexRetriever = SimstringIndexRetriever(
            index_path=os.path.join(kg.value.index_path, "classes_labels_index_simstring"),
            similarity=Similarity.LEVENSHTEIN,
            threshold=0.7,
        )
        self.class_faiss_retriever: FaissIndexRetriever = FaissIndexRetriever(
            index_path=os.path.join(kg.value.index_path, "classes_full_index_faiss"),
            threshold=0.7,
        )
        self.class_retriever = self.class_faiss_retriever
        self.entity_simstring_retriever: SimstringIndexRetriever = SimstringIndexRetriever(
            index_path=os.path.join(kg.value.index_path, "entities_labels_index_simstring"),
            similarity=Similarity.COSINE,
            threshold=0.75,
        )
        self.entity_retriever = self.entity_simstring_retriever
        self.property_simstring_retriever: SimstringIndexRetriever = SimstringIndexRetriever(
            index_path=os.path.join(kg.value.index_path, "predicates_labels_index_simstring"),
            similarity=Similarity.LEVENSHTEIN,
            threshold=0.7,
        )
        self.property_faiss_retriever: FaissIndexRetriever = FaissIndexRetriever(
            index_path=os.path.join(kg.value.index_path, "predicates_full_index_faiss"),
            threshold=0.7,
        )
        self.property_retriever = self.property_faiss_retriever 
            
    # ----------------------
    # ----- SMART FIND -----
    # ----------------------
    
    def smart_find_unified(self, query: str, return_urils: bool): # TODO: or queries
        return self.smart_find_separate(query, query, query, return_urils)
    
    def smart_find_separate(self, entity_query: str, class_query: str, property_query: str, return_urils: bool): # TODO: or queries
        return self.smart_find_multiple(
            entity_queries=[entity_query],
            class_queries=[class_query],
            property_queries=[property_query],
            return_urils=return_urils
        )
    
    def smart_find_multiple(self, entity_queries: list, class_queries: list, property_queries: list, return_urils: bool):
        all_classes = set()
        all_entities = set()
        all_properties = set()
        
        start_time = time()
        for cq in class_queries:
            if cq.strip() == "":
                continue
            classes = self.find_classes(cq)
            all_classes.update(classes)
        end_time = time()
        print(f"**Class retrieval time for {len(class_queries)} queries: {end_time - start_time:.2f} seconds.")
        
        start_time = time()
        for eq in entity_queries:
            if eq.strip() == "":
                continue
            entities = self.find_entities(eq)
            all_entities.update(entities)
        end_time = time()
        print(f"**Entity retrieval time for {len(entity_queries)} queries: {end_time - start_time:.2f} seconds.")
        
        start_time = time()
        for pq in property_queries:
            if pq.strip() == "":
                continue
            properties = self.find_properties(pq)
            all_properties.update(properties)
        end_time = time()
        print(f"**Property retrieval time for {len(property_queries)} queries: {end_time - start_time:.2f} seconds.")
        
        start_time = time()
        results = self.combine_results(list(all_classes), list(all_entities), list(all_properties))
        end_time = time()
        print(f"Combining results time: {end_time - start_time:.2f} seconds.")
        
        start_time = time()
        if return_urils:
            results['classes'] = self.kg.uris_to_urils(results['classes'])
            results['entities'] = self.kg.uris_to_urils(results['entities'])
            results['properties'] = self.kg.uris_to_urils(results['properties'])
            results['entities_predicates_map'] = {
                self.kg.uri_to_uril(entity) : [
                    self.kg.uri_to_uril(predicate) for predicate in predicates
                ] for entity, predicates in results['entities_predicates_map'].items()
            }
            results['classes_predicates_map'] = {
                self.kg.uri_to_uril(cls) : [
                    self.kg.uri_to_uril(predicate) for predicate in predicates
                ] for cls, predicates in results['classes_predicates_map'].items()
            }
        end_time = time()
        print(f"URI to URIL conversion time: {end_time - start_time:.2f} seconds.")
        
        return {'classes_predicates_map': results['classes_predicates_map'],
                'entities_predicates_map': results['entities_predicates_map'],}
        
    # -----------------------
    # ----- MANUAL FIND -----
    # -----------------------
        
    @lru_cache(maxsize=128)
    def get_predicates_for_entity(self, entity: str, endpoint: str, limit: int = 20):
        QUERY = f"""
            SELECT ?p (COUNT (?o) as ?c) WHERE {{
                <{entity}> ?p ?o .
            }}
            GROUP BY ?p
            ORDER BY DESC(?c)
            LIMIT {limit}
        """
        try:
            results = execute_sparql_query(QUERY, endpoint).convert()
        except Exception as e:
            print(f"Error querying SPARQL: {e}")
            print(f"Query: {QUERY}")
            return ['Error during processing, no predicates found.']
        predicates = []
        for result in results["results"]["bindings"]:
            predicates.append(result["p"]["value"])
        return predicates
    
    def _get_most_similar_predicates_for_entity(self, entity: str, search_query: str, threshold: float = 0.85):
        predicates_uris = self.get_predicates_for_entity(entity, self.kg.endpoint, 1024)
        predicates_labels = [self.kg.get_kg_component(uri).label for uri in predicates_uris]
        
        print(search_query)
        # print(predicates_labels)
        
        query_embedding = embed(search_query, is_query=False)
        embeddings = embed(predicates_labels, is_query=False)
        most_similar_predicates = []
        for i, predicate_embedding in enumerate(embeddings):
            similarity = cos_sim(query_embedding, predicate_embedding).item()
            if similarity >= threshold:
                print(f"Similar: {predicates_uris[i]} (label: {predicates_labels[i]}) with similarity {similarity:.4f}")
                most_similar_predicates.append((predicates_uris[i], similarity))
        most_similar_predicates = sorted(most_similar_predicates, key=lambda x: x[1], reverse=True)[:5]
        
        return most_similar_predicates
        
    def find_entity_with_properties(self, entity_query: str, properties_queries: List[str], fast: bool):
        if fast == False:
            candidate_entities = self.find_entities(entity_query)
            results = {}
            for e in candidate_entities:
                results[self.kg.uri_to_uril(e)] = []
                print("Finding predicates for entity: " + e)
                for prop_query in properties_queries:
                    print(" - Property query: " + prop_query)
                    properties = self._get_most_similar_predicates_for_entity(e, prop_query)
                    properties = [(self.kg.uri_to_uril(p), sim) for p, sim in properties]
                    print("   - Found properties:")
                    for p, sim in properties:
                        print(f"     - {p} (similarity: {sim:.4f})")
                    results[self.kg.uri_to_uril(e)].extend(properties)
                print("Entity " + e + " predicates found: " + str(len(results[self.kg.uri_to_uril(e)])))
                results[self.kg.uri_to_uril(e)] = list(set(results[self.kg.uri_to_uril(e)]))
            return results
        elif fast == True:
            results = self.smart_find_multiple([entity_query], "", properties_queries, return_urils=True)
            return results['entities_predicates_map']
        
    def get_predicates_for_class(self, clss: str) -> List[PredicateInfo]:
        try:
            classObj: KgClass = self.kg.get_kg_component(clss)
            return classObj.get_all_predicates(Direction.OUTGOING, StartPoint.CLASS)
        except Exception as e:
            print(f"Error getting predicates for class {clss}: {e}")
            return []
        
    def _get_most_similar_predicates_for_class(self, clss: str, search_query: str, threshold: float = 0.85):
        predicates = self.get_predicates_for_class(clss)
        predicates_uris = [p.uri for p in predicates]
        predicates_labels = [self.kg.get_kg_component(uri).label for uri in predicates_uris]
        
        print(search_query)
        # print(predicates_labels)
        
        query_embedding = embed(search_query, is_query=False)
        embeddings = embed(predicates_labels, is_query=False)
        most_similar_predicates = []
        for i, predicate_embedding in enumerate(embeddings):
            similarity = cos_sim(query_embedding, predicate_embedding).item()
            if similarity >= threshold:
                print(f"Similar: {predicates_uris[i]} (label: {predicates_labels[i]}) with similarity {similarity:.4f}")
                most_similar_predicates.append((predicates[i], similarity))
        most_similar_predicates = sorted(most_similar_predicates, key=lambda x: x[1], reverse=True)[:5]
        
        return most_similar_predicates
        
    def find_class_with_properties(self, class_query: str, properties_queries: List[str], fast: bool):
        if fast == False:
            candidate_classes = self.find_classes(class_query)
            results = {}
            for c in candidate_classes:
                results[self.kg.uri_to_uril(c)] = []
                print("Finding predicates for class: " + c)
                for prop_query in properties_queries:
                    print(" - Property query: " + prop_query)
                    properties = self._get_most_similar_predicates_for_entity(c, prop_query)
                    properties = [(self.kg.uri_to_uril(p), sim) for p, sim in properties]
                    print("   - Found properties:")
                    for p, sim in properties:
                        print(f"     - {p} (similarity: {sim:.4f})")
                    results[self.kg.uri_to_uril(c)].extend(properties)
                print("Class " + c + " predicates found: " + str(len(results[self.kg.uri_to_uril(c)])))
                results[self.kg.uri_to_uril(c)] = list(set(results[self.kg.uri_to_uril(c)]))
            return results
        elif fast == True:
            results = self.smart_find_multiple([], [class_query], properties_queries, return_urils=True)
            return results['classes_predicates_map']
    
    # ---------------------
    # ----- UTILITIES -----
    # ---------------------
    
    def calculate_boosted_score(self, similarity_score, popularity, type: KgComponentType, w=0.1):
        """
        Calculates a final score where popularity breaks ties but never dominates.
        
        Args:
            similarity_score (float): The relevance score (e.g., BM25, Cosine Similarity).
                                    Expected range: 0.0 to 1.0 (or normalized).
            popularity (int/float): The raw popularity metric (views, sales, etc.).
            w (float): Bandwidth. The maximum points popularity can add.
                    (e.g., 0.1 means popularity can add at most 0.1 to the score).
            k (int): Half-saturation constant. The popularity value that yields 
                    50% of the max boost.
                    
        Returns:
            float: The final boosted score.
        """
        if type == KgComponentType.ENTITY:
            k = self.kg.value.entity_popularity_k
        elif type == KgComponentType.CLASS:
            k = self.kg.value.class_popularity_k
        elif type == KgComponentType.PREDICATE:
            k = self.kg.value.predicate_popularity_k
        
        # Calculate the saturation boost (0.0 to 1.0)
        saturation = popularity / (popularity + k)
        
        # Scale it by the bandwidth weight
        boost = w * saturation
        
        return similarity_score + boost

    def find_entities(self, query: str, k: int = None, rank = True):
        return self._find(query=query, retriever=self.entity_retriever, k=k, rank=rank)
    
    def find_classes(self, query: str, k: int = None, rank = True):
        return self._find(query=query, retriever=self.class_retriever, k=k, rank=rank)
    
    def find_classes_simstring(self, query: str, k: int = None, rank = True):
        return self._find(query=query, retriever=self.class_simstring_retriever, k=k, rank=rank)
    
    def find_classes_faiss(self, query: str, k: int = None, rank = True):
        return self._find(query=query, retriever=self.class_faiss_retriever, k=k, rank=rank)
    
    def find_properties(self, query: str, k: int = None, rank = True):
        return self._find(query=query, retriever=self.property_retriever, k=k, rank=rank)
    
    def find_properties_simstring(self, query: str, k: int = None, rank = True):
        return self._find(query=query, retriever=self.property_simstring_retriever, k=k, rank=rank)
    
    def find_properties_faiss(self, query: str, k: int = None, rank = True):
        return self._find(query=query, retriever=self.property_faiss_retriever, k=k, rank=rank)
    
    def _find(self, query: str, retriever: IndexRetriever, k: int = None, rank = True):
        if k is None and self.k is not None:
            k = self.k
        elif k is None and self.k is None:
            raise ValueError("Either k parameter or Finder.k must be set.")
        candidates, scores = retriever.retrieve(query=query, debug=False, logging=False, include_labels=False, return_scores=True, k=k)
        if rank:
            # Boost scores with popularity
            boosted_candidates = []
            for cand, score in zip(candidates, scores):
                kgc = self.kg.get_kg_component(cand)
                if kgc is not None:
                    if kgc.type == KgComponentType.ENTITY:
                        popularity = kgc.incoming_edges_count + kgc.outgoing_edges_count
                    elif kgc.type == KgComponentType.CLASS:
                        popularity = kgc.incoming_edges_count + kgc.outgoing_edges_count
                    elif kgc.type == KgComponentType.PREDICATE:
                        popularity = kgc.incoming_edges_count + kgc.outgoing_edges_count
                    else:
                        popularity = 0
                else:
                    popularity = 0
                boosted_score = self.calculate_boosted_score(score, popularity, kgc.type if kgc is not None else KgComponentType.ENTITY)
                boosted_candidates.append((cand, boosted_score))
            # Re-rank by boosted score
            boosted_candidates = sorted(boosted_candidates, key=lambda x: x[1], reverse=True)
            candidates = [bc[0] for bc in boosted_candidates]
        return candidates[:k]
    

    def combine_results(self, classes, entities, properties):
        predicates_found = set()
        
        # Entities
        results = execute_sparql_query(SPARQL_MATCH_ENTITIES_WITH_PREDICATES.format(
            entities=" ".join(f"<{e}>" for e in entities),
            predicates=" ".join(f"<{p}>" for p in properties)
        ), self.kg.endpoint)
        results = results.convert()
        
        entities_predicates_map = {entity : [] for entity in entities}
        
        for r in results["results"]["bindings"]:
            entity_uri = r['entity']['value']
            predicate_uri = r['predicate']['value']
            entities_predicates_map[entity_uri].append(predicate_uri)
            predicates_found.add(predicate_uri)
            
        # Classes
        classes_predicates_map = {cls : [] for cls in classes}
        
        # if len(self.kg.value.get_kg_component(cls).get_predicate_uris(Direction.OUTGOING, KgClass.StartPoint.CLASS)) > 0:
        for cls in classes:
            if cls[0] != '<' and cls[-1] != '>':
                cls = f"<{cls}>"
            if self.kg.value.get_kg_component(cls) is None:
                print(f"Class {cls} not found in data.")
                continue
            clssObj: KgClass = self.kg.value.get_kg_component(cls)
            predicates_uris = clssObj.get_predicate_uris(Direction.OUTGOING, StartPoint.CLASS)
            predicates_found.update(predicates_uris)        

        predicates_not_found = set(properties) - predicates_found
        
        return {
            "classes": classes,
            "entities": entities,
            "properties": properties,
            "entities_predicates_map": entities_predicates_map,
            "classes_predicates_map": classes_predicates_map,
            "predicates_not_found": list(predicates_not_found)
        }