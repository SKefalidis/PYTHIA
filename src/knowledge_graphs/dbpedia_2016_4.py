from src.knowledge_graphs.knowledge_graph import KnowledgeGraph
import textwrap

class Dbpedia4(KnowledgeGraph):
    
    @property
    def prefixes(self):
        return textwrap.dedent("""
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX prov: <http://www.w3.org/ns/prov#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            PREFIX res: <http://dbpedia.org/resource/> 
            PREFIX dbp: <http://dbpedia.org/property/>
            PREFIX dct: <http://purl.org/dc/terms/> 
            PREFIX dbc: <http://dbpedia.org/resource/Category:>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
            PREFIX onto: <http://dbpedia.org/ontology/>
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX dbr: <http://dbpedia.org/resource/>
            PREFIX yago: <http://yago-knowledge.org/resource/>
        """)
        
    @property
    def entity_popularity_k(self) -> int:
        "Normally set to the median, we set it higher to bias towards popular entities."
        return 141
    
    @property
    def class_popularity_k(self) -> int:
        "Median popularity of classes."
        return 2674
    
    @property
    def predicate_popularity_k(self) -> int:
        "Median popularity of predicates."
        return 10