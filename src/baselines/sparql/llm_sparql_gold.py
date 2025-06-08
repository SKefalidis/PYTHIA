# Using LLMs to generate SPARQL queries for the given question, with gold entities, classes and predicates.
import textwrap
from typing import List, Tuple
from typing_extensions import override
from src.knowledge_graphs.knowledge_graph import KnowledgeGraph
from src.baselines.sparql.llm_sparql import LlmSparqlBaseline
from src.engine.qa.query_generator.query_db import QueryDb
from src.engine.class_identifier.gold_class_identifier import GoldClassIdentifier
from src.engine.entity_linking.gold_entity_identifier import GoldEntityLinker
from src.engine.predicate_linking.gold_predicate_identifier import GoldPredicateIdentifier
from src.metrics import PerformanceMetrics


class LlmSparqlGoldBaseline(LlmSparqlBaseline):    
    
    def __init__(self, model: str, kg_name: str, use_cot: bool, use_few_shot: bool, query_db: QueryDb,
                 kg: KnowledgeGraph, gold_entity_identifier: GoldEntityLinker, gold_class_identifier: GoldClassIdentifier, gold_predicate_identifier: GoldPredicateIdentifier,
                 use_gold_entities: bool = True, use_gold_classes: bool = True, use_gold_predicates: bool = True):
        super().__init__(model, kg_name, kg, use_cot, use_few_shot, query_db)
        self.kg = kg
        self.gold_entity_identifier = gold_entity_identifier
        self.gold_class_identifier = gold_class_identifier
        self.gold_predicate_identifier = gold_predicate_identifier

    # @override
    # def generate_sparql(self, question, **kwargs) -> Tuple[str, PerformanceMetrics]:
    #     query, messages, metrics = super().generate_sparql(question, **kwargs)
    #     # Replace URILs with URIs
    #     query = self.kg.triples_with_urils_to_triples_with_uris(query)
    #     return query, messages, metrics
    
    @override
    def _system_prompt(self):
        return super()._system_prompt() + "\n- You have access to the gold entities, classes, and predicates relevant to the question. These are sufficient for writing a query. Do not use any other entities, classes, or predicates."
    
    @override
    def _user_prompt(self, question: str, **kwargs):
        query = kwargs.get('query', None)
        
        gold_entities = self.gold_entity_identifier.identify(query)
        gold_entities_labels = [self.kg.get_label_from_uri(entity) for entity in gold_entities]
        # gold_entities = self.kg.uris_to_urils(gold_entities)
        gold_classes = self.gold_class_identifier.identify(query)
        gold_classes_labels = [self.kg.get_label_from_uri(cls) for cls in gold_classes]
        # gold_classes = self.kg.uris_to_urils(gold_classes)
        gold_predicates = self.gold_predicate_identifier.identify(query)
        gold_predicates_labels = [self.kg.get_label_from_uri(pred) for pred in gold_predicates]
        # gold_predicates = self.kg.uris_to_urils(gold_predicates)
        
        gold_entities_str = ', '.join([f"{label} (<{entity}>)" for label, entity in zip(gold_entities_labels, gold_entities)]) if gold_entities else "None"
        gold_classes_str = ', '.join([f"{label} (<{cls}>)" for label, cls in zip(gold_classes_labels, gold_classes)]) if gold_classes else "None"
        gold_predicates_str = ', '.join([f"{label} (<{pred}>)" for label, pred in zip(gold_predicates_labels, gold_predicates)]) if gold_predicates else "None"
        
        return textwrap.dedent(
            f"""
                Generate a SPARQL query for the following question:
                Question: {question}
                Gold Entities: {gold_entities_str}
                Gold Classes: {gold_classes_str}
                Gold Predicates: {gold_predicates_str}
                SPARQL Query:
                """)
        
    @override
    def _few_shot_user_prompt(self, question: str, examples: List[Tuple[str, str]], **kwargs):
        examples_string = "\n" + '\n'.join([f"Question: {ex[0]}\nSPARQL Query:\n```{ex[1]}```" for ex in examples if len(ex) >= 2]) + "\n"
        user_prompt = self._user_prompt(question, **kwargs)
        return textwrap.dedent(
            f"""I want you to generate a SPARQL query for the following question: {question}
                Here are some examples to guide you:
                {examples_string}
                {user_prompt}""")