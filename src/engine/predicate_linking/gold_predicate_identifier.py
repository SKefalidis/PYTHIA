from src.datasets.geoquestions1089_dataset import Geoquestions1089Dataset
from src.utils import execute_sparql_query, get_relative_path
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.engine.gost_requests import extract_predicates


class GoldPredicateIdentifier():
    
    PREDICATES_TO_IGNORE = [
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.wikidata.org/prop/direct/P31"
    ]
    
    def __init__(self, prefixes, filter_predicates: bool):
        self.prefixes = prefixes
        if filter_predicates:
            self.predicates_to_ignore = self.PREDICATES_TO_IGNORE
        else:
            self.predicates_to_ignore = []
            
    def identify(self, query: str):        
        all_predicates = extract_predicates(self.prefixes + "\n" + query)
        if all_predicates == "" or all_predicates is None:
            return []
        all_predicates = all_predicates.split("\n")
        all_predicates = [p for p in all_predicates if p.strip() != '' and p not in self.predicates_to_ignore]
        return list(set(all_predicates))
    
    def get_name(self):
        return "Gold Predicate Identifier"
    
    def get_resource(self):
        return "GoST"


if __name__ == '__main__':
    prefixes = Geoquestions1089Dataset.get_prefixes()
    
    identifier = GoldPredicateIdentifier(prefixes, True)
    
    query = "SELECT DISTINCT ?forest ?canal WHERE { yago:Edinburgh geo:hasGeometry ?geomLC . ?geomLC geo:asWKT ?lcnWKT . ?forest rdf:type y2geoo:OSM_forest ; geo:hasGeometry ?geomL . ?geomL geo:asWKT ?lWKT . ?canal rdf:type y2geoo:OSM_canal ; geo:hasGeometry ?geomLC2 . ?geomLC2 geo:asWKT ?lcnWKT2 FILTER ( geof:sfWithin(?lWKT, ?lcnWKT) && geof:sfWithin(?lcnWKT2, ?lcnWKT) ) }"
    classes = identifier.identify(query)
    print(classes)