import json
from enum import Enum
from typing import Tuple, Any

from src.engine.agent.tools.query_execution import QueryExecutionTool
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.engine.gost_requests import validate_query_with_errors


class FinalQueryExecutionTool(QueryExecutionTool):    
    @classmethod
    def name(cls) -> str:
        return "query_execution_final"
    
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "execute_query",
                "description": "Execute a SPARQL query and retrieve its results if successful. If not, an error message is returned. \
                Use this only when you are ready to generate the final answer, to validate its results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The SPARQL query to execute."},
                    },
                    "required": ["query"],
                },
            },
        }