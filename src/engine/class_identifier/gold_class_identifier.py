from operator import index
# from src.datasets.geoquestions1089_dataset import Geoquestions1089Dataset
from src.utils import execute_sparql_query, get_relative_path
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.knowledge_graphs.knowledge_graph import KgComponent
from src.engine.class_identifier.class_identifier import ClassIdentifier
from src.engine.gost_requests import extract_uris
import os
import pickle


SPARQL_GET_CLASSES = """
    SELECT DISTINCT ?class
    WHERE {
        ?instance a ?class .
    }
"""

SPARQL_GET_CLASSES_WIKIDATA = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT DISTINCT ?class
    WHERE {
        ?instance wdt:P31 ?class .
        ?class <http://www.wikidata.org/prop/direct/P279> ?superclass .
    }
"""


class GoldClassIdentifier(ClassIdentifier):
    
    def __init__(self, knowledge_graph: KnowledgeGraphs, endpoint_url: str, prefixes: str):
        super().__init__()
        self.knowledge_graph = knowledge_graph
        self.endpoint_url = endpoint_url
        self.prefixes = prefixes
        
        print(f"Loading classes for {knowledge_graph}...")
        
        classes = []
        classes_cache_filepath = get_relative_path("./resources/knowledge_graph_classes/" + knowledge_graph.name.lower() + "_classes.pkl")
        if os.path.exists(classes_cache_filepath):
            try:
                with open(classes_cache_filepath, 'rb') as f:
                    self.classes = pickle.load(f)
                    return
            except Exception as e:
                print(f"Failed to load classes from cache: {e}")
        else:
            index_path = input("Enter the path to the knowledge graph index (or press Enter to skip): ").strip()
            if index_path.strip() != "":
                self.knowledge_graph.load(index_path)
                for component in self.knowledge_graph.value.kg_components.values():
                    if component.type == KgComponent.Type.CLASS:
                        classes.append(component.uri)
            else:
                # SPARQL Query to get all classes
                if knowledge_graph == KnowledgeGraphs.WIKIDATA:
                    query = SPARQL_GET_CLASSES_WIKIDATA
                else:
                    query = SPARQL_GET_CLASSES
                try:
                    ret = execute_sparql_query(query, endpoint_url, max_wait_minutes=100).convert()

                    for r in ret["results"]["bindings"]:
                        c = r['class']['value']
                        # print(c)
                        classes.append(c)
                except Exception as e:
                    print(e)
                self.classes = classes

            if len(classes) < 10:
                raise Exception("CRITICAL: Loaded less than 10 classes, something is probably wrong.")
            else:
                print(f"Found {len(classes)} classes.")
            
            # Save to disk
            try:
                with open(classes_cache_filepath, 'wb') as f:
                    print("Saving classes to cache...")
                    pickle.dump(classes, f)
            except Exception as e:
                print(f"Failed to save classes to cache: {e}")

            self.classes = classes

        
    def identify(self, query: str):
        """
        Given a question, identifies all the relevant classes in YAGO2GEO.
        
        :param question: The natural language question to identify classes from.
        :return: A list of relevant classes in YAGO2GEO.
        """
        
        # print(self.prefixes + "\n" + query)
        
        all_uris = extract_uris(self.prefixes + "\n" + query)
        # print(all_uris)
        if all_uris == "" or all_uris is None:
            return []
        all_uris = all_uris.split("\n")
        class_uris = []
        for u in all_uris:
            if u in self.classes:
                class_uris.append(u)
            # else:
            #     print(f"URI {u} not in classes")
        # print(class_uris)
        return list(set(class_uris))
    
    def get_name(self):
        return self.knowledge_graph + " Gold Class Identifier"
    
    def get_resource(self):
        return "GoST"


# if __name__ == '__main__':
#     prefixes = Geoquestions1089Dataset.get_prefixes()
    
#     identifier = GoldClassIdentifier(KnowledgeGraphs.YAGO2geo, KnowledgeGraphs.get_endpoint(KnowledgeGraphs.YAGO2geo), prefixes)
    
#     query = "SELECT DISTINCT ?forest ?canal WHERE { yago:Edinburgh geo:hasGeometry ?geomLC . ?geomLC geo:asWKT ?lcnWKT . ?forest rdf:type y2geoo:OSM_forest ; geo:hasGeometry ?geomL . ?geomL geo:asWKT ?lWKT . ?canal rdf:type y2geoo:OSM_canal ; geo:hasGeometry ?geomLC2 . ?geomLC2 geo:asWKT ?lcnWKT2 FILTER ( geof:sfWithin(?lWKT, ?lcnWKT) && geof:sfWithin(?lcnWKT2, ?lcnWKT) ) }"
#     classes = identifier.identify(query)
#     print(classes)