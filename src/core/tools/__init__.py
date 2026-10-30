from src.core.tools.tool import Tool
from src.core.tools.graph_exploration.graph_search import GraphSearchTool
from src.core.tools.graph_exploration.stepwise_search import StepwiseSearchTool
from src.core.tools.direct_retrieval.find_anchors import FindAnchorsTool
from src.core.tools.query_execution.query_execution import QueryExecutionTool
from src.core.tools.graph_exploration.get_predicates import PredicatesTool
from src.core.tools.direct_retrieval.index_retrievers.finder import Finder
# from src.engine.agent.tools.query_execution_final import FinalQueryExecutionTool

from enum import Enum

class AvailableTools(Enum):
    FIND_ANCHORS = FindAnchorsTool.name()
    STEPWISE_SEARCH = StepwiseSearchTool.name()
    GRAPH_SEARCH = GraphSearchTool.name()
    EXECUTE_QUERY = QueryExecutionTool.name()
    GET_PREDICATES = PredicatesTool.name()