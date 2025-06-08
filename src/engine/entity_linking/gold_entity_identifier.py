from typing import List
from src.datasets.cwq_dataset import CwqDataset
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.engine.entity_linking.entity_linker import EntityLinker
from src.engine.class_identifier.gold_class_identifier import GoldClassIdentifier
from src.engine.gost_requests import extract_uris
from src.utils import execute_sparql_query


def generic_uri_is_entity(uri: str, endpoint: str) -> bool:
    # Is the URI used as a predicate?
    try:
        result = execute_sparql_query(f"""
            ASK WHERE {{
                ?s <{uri}> ?o .
            }}
        """, endpoint)
        if result.convert()['boolean']:
            return False 
    except Exception as e:
        print(f"Error checking if URI is a predicate: {e}")
        return False

    # Is the URI used as an rdf:type?
    try:
        result = execute_sparql_query(f"""
            ASK WHERE {{
                ?s a <{uri}> .
            }}
        """, endpoint)
        if result.convert()['boolean']:
            return False
    except Exception as e:
        print(f"Error checking if URI is an rdf:type: {e}")
        return False
    
    # print(f"URI: {uri} is not a predicate or rdf:type")
    
    return True

def elections_uri_is_entity(uri: str) -> bool:
    return y2geo_uri_is_entity(uri)

def stelar_uri_is_entity(uri: str) -> bool:
    return generic_uri_is_entity(uri, (KnowledgeGraphs.STELAR_KG.endpoint))

def terraq_uri_is_entity(uri: str) -> bool:
    return y2geo_uri_is_entity(uri)

def beastiary_uri_is_entity(uri: str) -> bool:
    return generic_uri_is_entity(uri, (KnowledgeGraphs.BEASTIARY_KG.endpoint))

def y2geo_uri_is_entity(uri: str) -> bool:
    return "/resource/" in uri and '/has' not in uri

def wikidata_uri_is_entity(uri: str) -> bool:
    return "http://www.wikidata.org/entity/" in uri # Should be correct, since we filter out classes with the GoldClassIdentifier.

def dbpedia_uri_is_entity(uri: str) -> bool:
    return "http://dbpedia.org/resource" in uri

def freebase_uri_is_entity(uri: str) -> bool:
    # print(f"URI: {uri}")
    if generic_uri_is_entity(uri, (KnowledgeGraphs.FREEBASE.endpoint)) == False:
        return False
    return "http://rdf.freebase.com/ns/" in uri


class GoldEntityLinker(EntityLinker):
    
    def __init__(self, knowledge_graph: str, prefixes: str):
        super().__init__(knowledge_graph)
        self.prefixes = prefixes
        self.class_identifier = None
        # if self.knowledge_graph == KnowledgeGraphs.YAGO2geo:
        #     self.entity_identification_function = y2geo_uri_is_entity
        if self.knowledge_graph == KnowledgeGraphs.WIKIDATA:
            self.entity_identification_function = wikidata_uri_is_entity
            self.class_identifier = GoldClassIdentifier(KnowledgeGraphs.WIKIDATA, KnowledgeGraphs.WIKIDATA.endpoint, prefixes)
        elif self.knowledge_graph == KnowledgeGraphs.DBPEDIA or self.knowledge_graph == KnowledgeGraphs.DBPEDIA10:
            self.entity_identification_function = dbpedia_uri_is_entity
        elif self.knowledge_graph == KnowledgeGraphs.FREEBASE:
            self.entity_identification_function = freebase_uri_is_entity
        # elif self.knowledge_graph == KnowledgeGraphs.ELECTIONS_KG:
        #     self.entity_identification_function = y2geo_uri_is_entity
        # elif self.knowledge_graph == KnowledgeGraphs.STELAR_KG:
        #     self.entity_identification_function = stelar_uri_is_entity
        # elif self.knowledge_graph == KnowledgeGraphs.TERRAQ_KG:
        #     self.entity_identification_function = terraq_uri_is_entity
        elif self.knowledge_graph == KnowledgeGraphs.BEASTIARY_KG:
            self.entity_identification_function = beastiary_uri_is_entity
    
    def nerd(self, query: str):
        all_uris = extract_uris(self.prefixes + "\n" + query)
        if all_uris is None:
            return []
        all_uris = all_uris.split("\n")
        
        if self.class_identifier is not None:
            class_uris = self.class_identifier.identify(query)
            all_uris = [u for u in all_uris if u not in class_uris]
        
        entity_uris = []
        for u in all_uris:
            if u == "":
                continue
            if self.entity_identification_function(u) == True:
                entity_uris.append(u)
        return list(set(entity_uris))
    
    def supported_targets(self) -> List[KnowledgeGraphs]:
        return [kg for kg in KnowledgeGraphs] 
    
    def get_name(self):
        return self.knowledge_graph + " Gold Entity Linker"
    
    def get_resource(self):
        return "GoST"
    
    
if __name__ == "__main__":
    # Test the GoldEntityLinker
    gold_entity_linker = GoldEntityLinker(KnowledgeGraphs.FREEBASE, CwqDataset.get_prefixes())
    query = """
    PREFIX ns: <http://rdf.freebase.com/ns/> 
    SELECT DISTINCT ?x WHERE { 
        FILTER (?x != ?c) FILTER (!isLiteral(?x) || lang(?x) = '' || langMatches(lang(?x), 'en')) 
        ?c ns:organization.organization.leadership ?k . ?k ns:organization.leadership.person ns:m.0hhv_6h .  
        ?c ns:sports.sports_team.championships ?x . ?x ns:time.event.start_date ?sk0 . 
    } 
    ORDER BY DESC(xsd:datetime(?sk0)) LIMIT 1
    """
    print(gold_entity_linker.nerd(query))