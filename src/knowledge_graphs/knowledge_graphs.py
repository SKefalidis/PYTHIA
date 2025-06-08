from enum import Enum
from src.knowledge_graphs.knowledge_graph import KgComponent
from src.knowledge_graphs.wikidata import Wikidata
from src.knowledge_graphs.freebase import Freebase
from src.knowledge_graphs.beastiary_kg import BeastiaryKg
from src.knowledge_graphs.dbpedia_2016_4 import Dbpedia4
from src.knowledge_graphs.dbpedia_2016_10 import Dbpedia10
from src.knowledge_graphs.custom_kg import CustomKg

    
class KnowledgeGraphs(Enum):
    """A controller object for handling KG endpoints and URIL/URI conversions. Enums are singletons. We have as many instances as KGs."""
    FREEBASE = Freebase()
    DBPEDIA = Dbpedia4()
    DBPEDIA10 = Dbpedia10()
    WIKIDATA = Wikidata()
    BEASTIARY_KG = BeastiaryKg()
    # CUSTOM = CustomKg() # For any other KG, not specifically supported.
    
    @property
    def endpoint(self):
        return self.value.endpoint
    
    @property
    def prefixes(self):
        return self.value.prefixes
    
    def get_kg_component(self, uri: str) -> KgComponent | None:
        return self.value.get_kg_component(uri)
    
    def load(self, knowledge_graph_index_path: str):
        self.value.load(knowledge_graph_index_path)
    
    def uri_to_uril(self, uri: str):# -> Any:
        return self.value.uri_to_uril(uri)

    def uris_to_urils(self, uris: list):
        return self.value.uris_to_urils(uris)

    def uril_to_uri(self, uril: str):
        return self.value.uril_to_uri(uril)
        
    def urils_to_uris(self, urils: list,):
        return self.value.urils_to_uris(urils)

    def triples_with_urils_to_triples_with_uris(self, triples: str):
        return self.value.triples_with_urils_to_triples_with_uris(triples)

    def triples_with_uris_to_triples_with_urils(self, triples: str):
        return self.value.triples_with_uris_to_triples_with_urils(triples)
    
    def get_label_from_uri(self, uri: str) -> str:
        return self.value.get_label_from_uri(uri)