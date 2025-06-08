import requests
from typing import List
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.engine.entity_linking.entity_linker import EntityLinker


class Bela(EntityLinker):

    def __init__(self, knowledge_graph: KnowledgeGraphs):
        super().__init__(knowledge_graph)
        
    def nerd(self, question: str, debug: bool = False, logging: bool = False):
        results = requests.post("http://localhost:8001/nerd", json={"question": question}).json()['result']
        
        uris = []
        for entry in results:
            for entity in entry['entities']:
                uris.append(f"http://www.wikidata.org/entity/{entity}")
        
        if self.convert:
            uris = self.convert_to_kg(uris)      
            
        # print("Question:", question)
        # print("BELA found entities:", uris)  
        
        if logging == False:
            return uris
        else:
            return uris, None
    
    def get_name(self):
        return "BELA"
    
    def supported_targets(self) -> List[KnowledgeGraphs]:
        return [KnowledgeGraphs.WIKIDATA]


if __name__ == '__main__':
    entity_linker = Bela(KnowledgeGraphs.WIKIDATA)
    entities = entity_linker.nerd(
        question="Which counties in California were won by the Republican party?")
    print(entities)
