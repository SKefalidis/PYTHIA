from textwrap import dedent
from typing import Any
from src.engine.agent.tools.tool import Tool
from src.engine.index_retrievers.finder import Finder
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs

class FindAnchorsTool(Tool):
    
    def __init__(self, finder: Finder, kg: KnowledgeGraphs, k: int = 20):
        super().__init__()
        self.finder = finder
        self.kg = kg.value
        self.k = k
        self.returned_entities = set()
        self.returned_classes = set()
    
    @classmethod
    def name(self) -> str:
        return "retrieve_entities_and_classes"
    
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "retrieve_entities_and_classes",
                "description": "This tool searches knowledge indices to find nodes that match the given label. Use it to find named entities and classes in the knowledge graph.\
                    e.g., if the label is 'Albert Einstein', it should return the entity corresponding to Albert Einstein in the knowledge graph.\
                    Note that this tool does not return relations or predicates, only entities and classes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "The label of the node to search for in the knowledge graph."},
                    },
                    "required": ["label"],
                },
            },
        }
    
    def function(self, *args, **kwargs) -> Any:
        label = kwargs.get("label")
        # description = kwargs.get("description")
        if label is None or label.strip() == "":
            return "An empty label was provided to `retrieve_entities_and_classes`. Please provide a label as string."
        entities_uris = self.finder.find_entities(label, k=self.k)
        entities_components = [self.kg.get_kg_component(e) for e in entities_uris]
        # print(list(zip(entities_uris, entities_components)))
        classes_uris = self.finder.find_classes(label, k=self.k)
        classes_components = [self.kg.get_kg_component(c) for c in classes_uris]
        # print(list(zip(classes_uris, classes_components)))
        
        self.returned_entities = self.returned_entities.union(set(entities_uris))
        self.returned_classes = self.returned_classes.union(set(classes_uris))
        
        return {
            "results": {
                "entities": [ 
                    {"uri": e, "label": kgc.label, "description": kgc.description[:100], "popularity": kgc.incoming_edges_count+kgc.outgoing_edges_count, "types": kgc.parent_classes} for e, kgc in zip(entities_uris, entities_components) if kgc is not None
                ],
                "classes": [
                    {"uri": c, "label": kgc.label, "description": kgc.description[:100], "popularity": kgc.incoming_edges_count+kgc.outgoing_edges_count, "superclasses": kgc.parent_classes} for c, kgc in zip(classes_uris, classes_components) if kgc is not None
                ]
            }
        }
        
        
if __name__ == "__main__":
    import os
    from src.engine.config import CONFIG
    from src.engine.index_retrievers.finder import Finder
    from src.logging import create_console_logger
    
    create_console_logger()
    
    kg = KnowledgeGraphs.BEASTIARY_KG
    kg.load(os.path.join(CONFIG().get("index_dir"), "beastiary"))
    
    finder = Finder(kg)
    
    tool = FindAnchorsTool(finder, kg)
    
    result = tool.function(label="draconic language")
    print("Search results for label 'Draconic Language':")
    for entity in result['results']['entities']:
        print(f"Entity URI: {entity['uri']}, Label: {entity['label']}, Description: {entity['description']}, Popularity: {entity['popularity']}, Types: {entity['types']}")
    for cls in result['results']['classes']:
        print(f"Class URI: {cls['uri']}, Label: {cls['label']}, Description: {cls['description']}, Popularity: {cls['popularity']}, Superclasses: {cls['superclasses']}")