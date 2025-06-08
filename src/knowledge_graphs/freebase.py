from typing_extensions import override, Optional
from src.knowledge_graphs.knowledge_graph import KnowledgeGraph
from src.utils import execute_sparql_query
import textwrap


class Freebase(KnowledgeGraph):
    
    @property
    def prefixes(self):
        return textwrap.dedent("""
            PREFIX uom: <http://www.opengis.net/def/uom/OGC/1.0/>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            PREFIX ns: <http://rdf.freebase.com/ns/>
        """)
        
    @property
    def entity_popularity_k(self) -> int:
        "Normally set to the median, we set it higher to bias towards popular entities."
        return 52
    
    @property
    def class_popularity_k(self) -> int:
        "Median popularity of classes."
        return 27
    
    @property
    def predicate_popularity_k(self) -> int:
        "Median popularity of predicates."
        return 2
    
    @override
    def register_uri_to_uril_mapping(self, uri: str):
        id = uri.split("/")[-1]
        if len(id) > 2 and (id[1] == '.' or id[2] == '.'):
            kg_component = self.get_kg_component(uri)
            if kg_component is not None:
                label = self.get_kg_component(uri).label
            else: # Fallback: try to get label from Freebase
                label = self.get_freebase_label(uri)
            if label is not None :
                label = label.replace(" ", "_")
                self.uri_to_uril_map[uri] = uri + "_" + label
                self.uril_to_uri_map[uri + "_" + label] = uri
            else:
                self.uri_to_uril_map[uri] = uri
                self.uril_to_uri_map[uri] = uri
        else:
            self.uri_to_uril_map[uri] = uri
            self.uril_to_uri_map[uri] = uri
            
    def get_freebase_label(self, uri) -> Optional[str]:    
        query = f"""
        SELECT ?tailEntity WHERE {{
                <{uri}> <http://www.w3.org/2000/01/rdf-schema#label> ?tailEntity .
                FILTER (lang(?tailEntity) = "en")
        }}
        """    
        try:
            results = execute_sparql_query(query, self.endpoint).convert()
            if len(results["results"]["bindings"]) > 0:
                return results["results"]["bindings"][0]["tailEntity"]["value"]
            else:
                return None
        except Exception as e:
            print(f"Error: {e}")
            return "Unnamed Entity"
        

if __name__ == "__main__":
    uri = "http://rdf.freebase.com/ns/m.03_r3"
    
    label = Freebase.get_freebase_label(uri)
    
    print(f"Label for {uri}: {label}")