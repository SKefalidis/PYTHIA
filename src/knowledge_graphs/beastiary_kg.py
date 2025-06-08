from src.knowledge_graphs.knowledge_graph import KnowledgeGraph


class BeastiaryKg(KnowledgeGraph):
    
    @property
    def prefixes(self):
        return """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX uom: <http://www.opengis.net/def/uom/OGC/1.0/>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX beast: <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#>
        """
        
    @property
    def entity_popularity_k(self) -> int:
        "Normally set to the median, we set it higher to bias towards popular entities."
        return 10
    
    @property
    def class_popularity_k(self) -> int:
        "Median popularity of classes."
        return 100
    
    @property
    def predicate_popularity_k(self) -> int:
        "Median popularity of predicates."
        return 10