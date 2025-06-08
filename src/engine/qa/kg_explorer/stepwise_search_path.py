from sre_constants import LITERAL
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum
from re import S, sub

from src.engine.qa.kg_explorer.kg_path import KgPath
from src.knowledge_graphs.knowledge_graph import Direction, KnowledgeGraph
from src.knowledge_graphs.knowledge_graph import KnowledgeGraph, PredicateInfo, Direction


# -------------------------------------------
# ----- Classes used in Stepwise Search -----
# -------------------------------------------

class StepwiseSearchNodeType(Enum):
    ENTITY = 1
    LITERAL = 2
    MIXED = 3
    INTERMEDIATE = 4


@dataclass
class StepwiseSearchEdge:
    start_node: 'StepwiseSearchNode'
    predicate: PredicateInfo # we ignore objects
    predicate_label: str
    end_node: 'StepwiseSearchNode' = None


class StepwiseSearchNode:

    # -----------------------
    # ----- Constructor -----
    # -----------------------
    
    def __init__(self, value: str|List[str], label: str|List[str]|None, previous: StepwiseSearchEdge|None, target_description: str):
        # Node information
        self.value: List[str] = value if isinstance(value, List) else [value]
        has_literal = False
        has_uri = False
        for v in self.value:
            if "http:" in v:
                has_uri = True
            else:
                has_literal = True
        if has_literal and has_uri:
            self.node_type = StepwiseSearchNodeType.MIXED
        elif has_literal:
            self.node_type = StepwiseSearchNodeType.LITERAL
        else:
            self.node_type = StepwiseSearchNodeType.ENTITY
        self.label: List[str] = label if isinstance(label, List) else ([label] if label is not None else ["Unnamed Entity"] * len(self.value))
        is_unnamed = all(lab == "Unnamed Entity" for lab in self.label)
        if is_unnamed:
            self.node_type = StepwiseSearchNodeType.INTERMEDIATE

        if len(self.value) != len(self.label):
            raise ValueError("value and label must have the same length.")
        
        # Links to other nodes
        self.previous: StepwiseSearchEdge|None = previous
        if self.previous is not None:
            self.previous.end_node = self
        self.next: List[StepwiseSearchEdge] = []
        
        # Search information
        self._target_description = target_description # If set, we use it for diverging search paths.
        self._predicates: List[PredicateInfo] = []
        self._frontier_predicates: List[PredicateInfo] = []
        self._followed_predicates: List[PredicateInfo] = [] # predicates that have been already explored from this node.
        
    def generate_successor_by_expanding_predicate(self, predicate: PredicateInfo, kg: KnowledgeGraph) -> 'StepwiseSearchNode':
        connection = StepwiseSearchEdge(start_node=self, predicate=predicate, predicate_label=predicate.get_label(kg=kg))
        self.next.append(connection)
        objects = predicate.objects[:3] if len(predicate.objects) > 3 else predicate.objects
        object_labels = []
        for o in objects:
            kgc = kg.get_kg_component(o)
            if kgc is not None:
                object_labels.append(kgc.label)
            else:
                object_labels.append("Unnamed Entity") # fallback
        new_node = StepwiseSearchNode(objects, object_labels, connection, target_description=self._target_description)
        self._followed_predicates.append(predicate)
        return new_node
    
    def generate_successor_by_expanding_frontier_predicate(self, index: int, kg: KnowledgeGraph) -> 'StepwiseSearchNode':
        predicate = self.get_frontier_predicate_by_index(index)
        return self.generate_successor_by_expanding_predicate(predicate, kg)
        
    # ---------------------
    # ----- Accessors -----
    # ---------------------
        
    def set_predicates(self, predicates: List[PredicateInfo]):
        if not isinstance(predicates, List):
            raise ValueError("predicates must be a list of PredicateInfo objects.")
        if len(self._predicates) > 0:
            return
        self._predicates = predicates
        
    def get_predicates(self) -> List[PredicateInfo]:
        "Does not return predicates that have already been followed."
        predicates_to_return = []
        for predicate in self._predicates:
            if predicate not in self._followed_predicates:
                predicates_to_return.append(predicate)
        return predicates_to_return
    
    def set_frontier_predicates(self, frontier_predicates: List[PredicateInfo]):
        if not isinstance(frontier_predicates, List):
            raise ValueError("frontier_predicates must be a list of PredicateInfo objects.")
        if len(self._frontier_predicates) > 0:
            raise ValueError("frontier_predicates has already been set for this StepwiseSearchNode.")
        self._frontier_predicates = frontier_predicates
    
    def get_frontier_predicates(self) -> List[PredicateInfo]:
        return self._frontier_predicates
    
    def get_frontier_predicate_by_index(self, index: int) -> PredicateInfo:
        if index < 0 or index >= len(self._frontier_predicates):
            raise IndexError(f"{index} out of bounds for frontier predicates [0..{len(self._frontier_predicates)-1}]")
        return self._frontier_predicates[index]
        
    # ----------------------------------
    # ----- String Representations -----
    # ----------------------------------
    
    def get_textual_representation_of_value(self, readable: bool = False) -> str:
        value = self.value if not readable else self.label
        if isinstance(value, List):
            return ", ".join(value)
        elif isinstance(value, str):
            return value
        else:
            raise ValueError("value must be a string or a list of strings.")
    
    def full_path_representation(self, readable: bool = False, verbal: bool = False) -> str:
        # this node
        if verbal == False:
            open_symbol = "{"
            close_symbol = "}"
        else:
            open_symbol = "'"
            close_symbol = "'"
        
        if self.previous is None:
            return open_symbol + self.get_textual_representation_of_value(readable) + close_symbol
        
        predicate_str = self.previous.predicate.uri if not readable else self.previous.predicate_label
        if not verbal:
            if self.previous.predicate.direction == Direction.OUTGOING:
                connection_str = "-[:" + predicate_str + "]->"
            else:
                connection_str = "<-[: " + predicate_str + "]-"
        else:
            if self.previous.predicate.direction == Direction.OUTGOING:
                connection_str = " via '" + predicate_str + "' to "
            else:
                connection_str = " via '" + predicate_str + "' from "
        # previous node
        return self.previous.start_node.full_path_representation(readable, verbal) + connection_str + open_symbol + self.get_textual_representation_of_value(readable) + close_symbol
    
    def full_path_representation_with_frontier_predicate(self, predicate: PredicateInfo, kg=None, readable: bool = False, verbal: bool = False) -> str:
        base = self.full_path_representation(readable, verbal)
        if readable:
            predicate_str = predicate.get_label(kg)
        else:
            predicate_str = predicate.uri
        if verbal == False:
            if predicate.direction == Direction.OUTGOING:
                connection_str = "-[:" + predicate_str + "]->"
            else:
                connection_str = "<-[: " + predicate_str + "]-"
        else:
            if predicate.direction == Direction.OUTGOING:
                connection_str = " via '" + predicate_str + "' to "
            else:
                connection_str = " via '" + predicate_str + "' from "
        objects_str = predicate.get_objects_string(readable=True, kg=kg)
        return base + connection_str + "'" + objects_str + "'"
    
    def _predicate_path_representation(self, readable: bool = False, verbal: bool = False, include_start: bool = True) -> str:
        # this node
        if self.previous is None:
            if include_start:
                return "{" + self.get_textual_representation_of_value(readable) + "}"
            else:
                return ""
        
        predicate_str = self.previous.predicate.uri if not readable else self.previous.predicate_label
        if verbal == False:
            if self.previous.predicate.direction == Direction.OUTGOING:
                connection_str = "-[:" + predicate_str + "]->"
            else:
                connection_str = "<-[: " + predicate_str + "]-"
        else:
            if self.previous.predicate.direction == Direction.OUTGOING:
                connection_str = " following '" + predicate_str + "' "
            else:
                connection_str = " coming from '" + predicate_str + "' "

        # previous node
        return self.previous.start_node._predicate_path_representation(readable, verbal, include_start) + connection_str
    
    def predicate_path_representation(self, readable: bool = False, verbal: bool = False, include_start: bool = True, include_end: bool = True) -> str:
        if self.previous is None:
            if include_start:
                return "{" + self.get_textual_representation_of_value(readable) + "}"
            else:
                return ""
        if include_end:
            return self._predicate_path_representation(readable, verbal, include_start) + "{" + self.get_textual_representation_of_value(readable) + "}"
        else:
            return self._predicate_path_representation(readable, verbal, include_start)
        
    def predicate_path_representation_with_frontier_predicate(self, predicate: PredicateInfo, kg=None, readable: bool = False, verbal: bool = False) -> str:
        base = self.predicate_path_representation(readable, verbal, False, False)
        if readable:
            predicate_str = predicate.get_label(kg)
        else:
            predicate_str = predicate.uri
        if verbal == False:
            if predicate.direction == Direction.OUTGOING:
                connection_str = "-[:" + predicate_str + "]->"
            else:
                connection_str = "<-[: " + predicate_str + "]-"
        else:
            if predicate.direction == Direction.OUTGOING:
                connection_str = " following '" + predicate_str + "' "
            else:
                connection_str = " coming from '" + predicate_str + "' "
        return base + connection_str
    
# ----------------------------------------------------------------------------
# ----- Class used to represent a path explored via stepwise search in a -----
# ----- knowledge graph.                                                 -----
# ----------------------------------------------------------------------------

class StepwiseSearchPath(KgPath):
    
    def __init__(self, kg: KnowledgeGraph, end_node: StepwiseSearchNode):        
        super().__init__(kg)
        
        self.edges: List[StepwiseSearchEdge] = []
        current_node = end_node
        while current_node.previous is not None:
            self.edges.insert(0, current_node.previous)
            current_node = current_node.previous.start_node
    
    def get_tuples(self, readable: bool = False) -> List[Tuple[str, str, str]]:
        """
        Returns a List of Tuples of URI/variable triples (subject, predicate, object) representing the path.
        
        e.g., [("http://example.org/A", "http://example.org/knows", "?x1"), ("?x1", "http://example.org/likes", "http://example.org/B")]
        
        Args:
            readable (bool): If True, returns human-readable labels instead of URIs where possible.
        Returns:
            List[Tuple[str, str, str]]: List of triples representing the path.
        """
        tuples = []
        current = None
        var_index = 0
        for idx, edge in enumerate(self.edges):
            start_node = edge.start_node
            end_node = edge.end_node
            if len(start_node.value) > 1:
                subject = current
            else:
                subject = start_node.get_textual_representation_of_value(readable)
            predicate = edge.predicate.uri if not readable else edge.predicate_label
            if len(end_node.value) > 1:
                object = "?x" + str(var_index)
                current = object
                var_index += 1
            else:
                if readable == False and idx == len(self.edges) - 1:
                    # if "http:" in end_node.value[0]:
                    #     object = end_node.value[0]
                    # else:
                    object = "?x" + str(var_index)
                else:
                    object = end_node.get_textual_representation_of_value(readable)
                
            if edge.predicate.direction == Direction.OUTGOING:
                tuples.append((subject, predicate, object))
            elif edge.predicate.direction == Direction.INCOMING:
                tuples.append((object, predicate, subject))
            else:
                raise ValueError("Cannot generate triples for path with unknown direction parts.")
        return tuples
    
    def get_cypher_string(self, readable: bool = False) -> str:
        return self.edges[-1].end_node.full_path_representation(readable=readable, verbal=False)
    
    def get_cypher_normalized_string(self, readable: bool = False) -> str:
        return self.edges[-1].end_node.full_path_representation(readable=readable, verbal=False)
    
    def get_predicates(self) -> List[str]:
        return [edge.predicate.uri for edge in self.edges]
    
    def get_predicate_labels(self) -> List[str]:
        return [edge.predicate_label for edge in self.edges]
    
    def get_predicate_directions(self) -> List[Direction]:
        return [edge.predicate.direction for edge in self.edges]