from src.engine.agent.tools.tool import Tool
from src.engine.agent.tools.graph_search import GraphSearchTool
from src.engine.agent.tools.stepwise_search import StepwiseSearchTool
from src.engine.agent.tools.find_anchors import FindAnchorsTool
from src.engine.agent.tools.query_execution import QueryExecutionTool
from src.engine.agent.tools.get_predicates import PredicatesTool
# from src.engine.agent.tools.query_execution_final import FinalQueryExecutionTool

from enum import Enum

class AvailableTools(Enum):
    FIND_ANCHORS = FindAnchorsTool.name()
    STEPWISE_SEARCH = StepwiseSearchTool.name()
    GRAPH_SEARCH = GraphSearchTool.name()
    EXECUTE_QUERY = QueryExecutionTool.name()
    GET_PREDICATES = PredicatesTool.name()