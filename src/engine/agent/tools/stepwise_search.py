from typing import Any
from src.engine.agent.tools.tool import Tool
from src.engine.qa.kg_explorer.stepwise_search_manager import StepwiseSearchManager, StepwiseSearchResultEnum
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.utils import is_uri

class StepwiseSearchTool(Tool):
    
    def __init__(self, kg: KnowledgeGraphs, llm: str):
        super().__init__()
        self.search_manager = StepwiseSearchManager(kg, llm)
        self.max_steps = 3
        
        self.start_given = []
    
    @classmethod
    def name(cls) -> str:
        return "beam_search"
    
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "beam_search",
                "description": (
                    "Explore the knowledge graph from a start node for a depth of up to 3 steps.\n"
                    "Useful when you have a known start entity and want to find information related to it.\n"
                    "It uses a beam search strategy to explore multiple paths in parallel, allowing for a broader search of the graph. It uses the user question to guide the search.\n"
                    "For example in the question 'What is the age of Albert Einstein's mother?', with an anchor node corresponding to 'Albert Einstein', but no other nodes you could execute:\n"
                    "beam_search(start='http://example.org/Albert_Einstein')\n"                    
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start": {
                            "type": "string", 
                            "description": "Start URI (entity or class) that is known (anchor point). Without angle brackets. E.g., 'http://example.org/Albert_Einstein'"
                        },
                    },
                    "required": ["start"],
                },
            },
        }
        
    def function(self, *args, **kwargs) -> Any:
        """Find paths between known start and unknown goal URIs."""
        question = kwargs.get("question")
        start = kwargs.get("start")
        # target = kwargs.get("target")
        target = question
        
        if start is None or target is None:
            return "Incorrect usage of beam_search: both 'start' and 'target' parameters must be provided."
        
        if not is_uri(start):
            expanded = self.search_manager.kg.expand_uri(start)
            if not is_uri(expanded):
                return "Incorrect usage of beam_search: start URI is invalid. Make sure to provide a full URI, without angle brackets."
            else:
                start = expanded
                
        self.start_given.append(start)

        search_enum, paths, metrics = self.search_manager.search(target, 
                                                                 expanded if not is_uri(start) else start,
                                                                 max_steps=self.max_steps,
                                                                 enable_backtracking=False,
                                                                 enable_limit=True)

        if search_enum == StepwiseSearchResultEnum.FOUND:
            message = "Search successful. Relevant paths found."
            uris_used = set()
            for idx, path in enumerate(paths):
                triples = path.get_triples_string(shorten_uris=True)
                if len(path.sample_values) == 0:
                    path.find_sample_values(k=3)
                sample_results = path.sample_values
                if len(sample_results) < 3:
                    triples += "\n- Results:"
                else:
                    triples += "\n- Sample Results:"
                for sidx, s in enumerate(sample_results):
                    triples += f"\n-- {sidx+1}: {s}"
                message += f"\n\nPath {idx+1}:\n{triples}"
                
                uris_used.update(path.get_uris_used())
                for sample in sample_results:
                    for key, value in sample.items():
                        if isinstance(value, str) and is_uri(value):
                            uris_used.add(value)
                
            uris_used_list = list(uris_used)
            message += "\n\nReadable labels for URIs used in the paths and samples:\n"          
            for uri in uris_used_list:
                label = self.search_manager.kg.get_label_from_uri(uri)
                message += f"- {self.search_manager.kg.shorten_uri(uri)}: {label}\n"
        elif search_enum == StepwiseSearchResultEnum.CANCEL or search_enum == StepwiseSearchResultEnum.DEAD_END_REACHED:
            message = "Search was unable to find relevant paths. This usually means that the information is was too far from the start node or the target query was too distant semantically.\n"
            message += "Recommended action: Either find a closer start node or rephrase the target query. If this has been attempted already, consider an alternative plan for answering the question."
        elif search_enum == StepwiseSearchResultEnum.MAX_STEPS_REACHED:
            message = "Search reached the maximum allowed steps without finding relevant paths.\n"
            message += "Recommended action: Provide a closer start node. If this has been attempted already, consider an alternative plan for answering the question."
        else:
            message = "Search failed due to an unexpected error. Please try again or consider an alternative plan for answering the question."

        return message
    
    
if __name__ == "__main__":
    from src.engine.config import CONFIG
    from src.elelem.provider import ProviderFactory
    from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
    from src.logging import create_console_logger
    import os
    
    create_console_logger()

    kg = KnowledgeGraphs.WIKIDATA
    kg.load(os.path.join(CONFIG().get("index_dir"), "wikidata"))

    llm_provider = CONFIG().get_litellm_model_endpoint()  # e.g., "openai/gpt-4.1-nano"
    tool = StepwiseSearchTool(kg, llm_provider)

    result = tool.function(
        start="http://www.wikidata.org/entity/Q937",  # Albert Einstein
        target="the birth date of Albert Einstein's mother"
    )
    print(result)