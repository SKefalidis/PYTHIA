from src.knowledge_graphs.knowledge_graph import KnowledgeGraph
import textwrap


class Corporate(KnowledgeGraph):

    @property
    def prefixes(self):
        return textwrap.dedent("""
            PREFIX bd: <http://www.bigdata.com/rdf#>
            PREFIX cc: <http://creativecommons.org/ns#>
            PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX prov: <http://www.w3.org/ns/prov#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            PREFIX dct: <http://purl.org/dc/terms/> 
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
            PREFIX pvoc: <http://ld.company.org/prod-vocab/>
            PREFIX pins: <http://ld.company.org/prod-instances/>
        """)
        
    @property
    def entity_popularity_k(self) -> int:
        "Normally set to the median, we set it higher to bias towards popular entities."
        return 10

    @property
    def class_popularity_k(self) -> int:
        "Median popularity of classes."
        return 20
    
    @property
    def predicate_popularity_k(self) -> int:
        "Median popularity of predicates."
        return 10