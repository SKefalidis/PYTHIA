from typing import Any, List
from src.utils import is_uri
from src.engine.agent.tools.tool import Tool
from src.engine.qa.kg_explorer.graph_search_manager import GraphSearchManager
from src.engine.qa.kg_explorer.graph_search_path import GraphSearchPath
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs

class GraphSearchTool(Tool):
    
    def __init__(self, kg: KnowledgeGraphs, llm: str):
        super().__init__()
        self.search_manager = GraphSearchManager(kg, llm)
        self.kg = kg
        self.start_end_given = []
        self.tuples_returned = []
    
    @classmethod
    def name(cls) -> str:
        return "bidirectional_bfs"
    
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "bidirectional_bfs",
                "description": "This functions returns predicate paths between known start and goal nodes in the knowledge graph.\
                    It is useful when both the start and goal entities/classes are known, and you want to explore how they are connected in the knowledge graph in a fast and reliable way.\
                    For example, in the question 'Who is the author of the Lord of the Rings?', with anchor nodes corresponding to 'Lord of the Rings' and 'Author' to find the author relation you would execute:\
                    bidirectional_bfs(start='http://example.org/Lord_of_the_Rings', goal='http://example.org/Author')",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "description": "Start entity/class URI."},
                        "goal": {"type": "string", "description": "Goal entity/class URI."},
                    },
                    "required": ["start", "goal"],
                },
            },
        }
    
    def function(self, *args, **kwargs) -> Any:
        """Find paths between known start and goal URIs.

        Expected output format (example): similar to stepwise_explore.
        """
        question = kwargs.get("question")
        start = kwargs.get("start")
        goal = kwargs.get("goal")
        # relation_description = kwargs.get("relation_description")
        relation_description =  question
        
        if goal is None or goal.strip() == "":
            return "Incorrect usage of bidirectional_bfs: goal URI is missing. bidirectional_bfs requires both start and goal URIs to be provided. It is used to find paths between known start and goal nodes in the knowledge graph."
    
        if not is_uri(start):
            expanded = self.search_manager.kg.value.expand_uri(start)
            if not is_uri(expanded):
                return "Incorrect usage of bidirectional_bfs: start URI is invalid. Make sure to provide a full URI, without angle brackets."
            else:
                start = expanded
                
        if not is_uri(goal):
            expanded = self.search_manager.kg.value.expand_uri(goal)
            if not is_uri(expanded):
                return "Incorrect usage of bidirectional_bfs: goal URI is invalid. Make sure to provide a full URI, without angle brackets."
            else:
                goal = expanded
                
        self.start_end_given.append( (start, goal) )
    
        paths: List[GraphSearchPath]
        metrics: Any
        paths, metrics = self.search_manager.graph_search(sentence=relation_description, start_uri=start, end_uri=goal, focus_on_recall=True)
        if len(paths) > 0:
            message = "Search successful. Relevant paths found."
            uris_used = set()
            for idx, path in enumerate(paths):
                triples = path.get_triples_string(shorten_uris=True)
                if len(path.sample_values) == 0:
                    path.find_sample_values(k=3)
                sample_results = path.sample_values
                if len(sample_results) < 3 and len(sample_results) > 0:
                    triples += "\n- Results:"
                elif len(sample_results) >= 3:
                    triples += "\n- Sample Results:"
                for sidx, s in enumerate(sample_results):
                    triples += f"\n-- {sidx+1}: {s}"
                message += f"\n\nPath {idx+1}:\n{triples}"
                
                self.tuples_returned.extend(path.get_tuples())
                
                uris_used.update(path.get_uris_used())
                for sample in sample_results:
                    for key, value in sample.items():
                        if isinstance(value, str) and is_uri(value):
                            uris_used.add(value)
                        
            uris_used_list = list(uris_used)
            message += "\n\nReadable labels for URIs used in the paths and samples:\n"          
            for uri in uris_used_list:
                label = self.search_manager.kg.get_label_from_uri(uri)
                message += f"- {self.search_manager.kg.value.shorten_uri(uri)}: {label}\n"
        else:
            has_class = self.kg.value.is_class(start) or self.kg.value.is_class(goal)
            if has_class:
                message = "Search included class leading to a larger corridor size, search cancelled. Re-try with a specific entity."
            else:
                message = "No paths found between the given start and goal nodes for the automatically determined depth."
        
        return message
    
    
if __name__ == "__main__":
    from src.engine.config import CONFIG
    from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
    from src.logging import create_console_logger
    import os
    
    create_console_logger()

    kg = KnowledgeGraphs.WIKIDATA
    kg.load(os.path.join(CONFIG().get("index_dir"), "wikidata"))

    llm = CONFIG().get_litellm_model_endpoint()
    tool = GraphSearchTool(kg, llm)

    result = tool.function(
        start="http://www.wikidata.org/entity/Q729",  # Albert Einstein
        goal="http://www.wikidata.org/entity/Q625657",
        relation_description="animal participated in a military operation with Australian Defence Forces"
    )
    print(result)