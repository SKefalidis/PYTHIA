from typing import List, Tuple
from abc import ABC, abstractmethod
from re import sub

from src.knowledge_graphs.knowledge_graph import Direction, KnowledgeGraph
from src.utils import execute_sparql_query, is_uri


class KgPath(ABC):
    
    def __init__(self, kg: KnowledgeGraph):
        if not isinstance(kg, KnowledgeGraph):
            raise ValueError("kg must be an instance of KnowledgeGraph")
        self.kg = kg
        self.sample_values = []
        
    @abstractmethod
    def get_tuples(self, readable: bool = False) -> List[Tuple[str, str, str]]:
        """
        Returns a List of Tuples of URI/variable triples (subject, predicate, object) representing the path.
        
        e.g., [("http://example.org/A", "http://example.org/knows", "?x1"), ("?x1", "http://example.org/likes", "http://example.org/B")]
        
        Args:
            readable (bool): If True, returns human-readable labels instead of URIs where possible.
        Returns:
            List[Tuple[str, str, str]]: List of triples representing the path.
        """
        pass
    
    def get_triples(self, readable: bool = False, shorten_uris: bool = False) -> List[str]:
        triples = []
        for triple_tuple in self.get_tuples(readable=False):
            subject = triple_tuple[0]
            if "http:" in subject and not subject.startswith("<"):
                if readable:
                    label = self.kg.get_label(subject)
                    if label is not None:
                        label = sub(r'\s+', '_', label)
                        label = f"_{label}"
                    else:
                        label = ""
                else:
                    label = ""
                if shorten_uris:
                    subject = self.kg.shorten_uri(subject) + label
                else:
                    subject = "<" + subject+label + ">"
            predicate = triple_tuple[1]
            if "http:" in predicate and not predicate.startswith("<"):
                if readable:
                    label = self.kg.get_label(predicate)
                    if label is not None:
                        label = sub(r'\s+', '_', label)
                        label = f"_{label}"
                    else:
                        label = ""
                else:
                    label = ""
                if shorten_uris:
                    predicate = self.kg.shorten_uri(predicate) + label
                else:
                    predicate = "<" + predicate+label + ">"
            obj = triple_tuple[2]
            if "http:" in obj and not obj.startswith("<"):
                if readable:
                    label = self.kg.get_label(obj)
                    if label is not None:
                        label = sub(r'\s+', '_', label)
                        label = f"_{label}"
                    else:
                        label = ""
                else:
                    label = ""
                if shorten_uris:
                    obj = self.kg.shorten_uri(obj) + label
                else:
                    obj = "<" + obj+label + ">"
            triple_string = f"{subject} {predicate} {obj} ."
            triples.append(triple_string)
        return triples
    
    def get_triples_string(self, readable: bool = False, shorten_uris: bool = False) -> str:
        return "\n".join(self.get_triples(readable=readable, shorten_uris=shorten_uris))
    
    @abstractmethod
    def get_cypher_string(self, readable: bool = False) -> str:
        pass
    
    @abstractmethod
    def get_cypher_normalized_string(self, readable: bool = False) -> str:
        pass
    
    @abstractmethod
    def get_predicates(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_predicate_labels(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_predicate_directions(self) -> List[Direction]:
        pass
    
    def get_predicate_path(self, readable: bool = False) -> str:
        "A string representation of the predicate path including directions."
        path_str = ""
        directions = self.get_predicate_directions()
        if readable:
            predicates = self.get_predicate_labels()
            for i in range(len(predicates)):
                direction = directions[i]
                predicate = predicates[i]
                if direction == Direction.OUTGOING:
                    path_str += f"following '{predicate}' "
                else:
                    path_str += f"coming from '{predicate}' "
        else:
            predicates = self.get_predicates()
            for i in range(len(predicates)):
                direction = directions[i]
                predicate = predicates[i]
                if direction == Direction.OUTGOING:
                    path_str += f"-[{predicate}]->"
                else:
                    path_str += f"<-[{predicate}]-"
                    
    def get_uris_used(self) -> List[str]:
        "Returns all URIs used in the path (subjects and objects)."
        uris = set()
        for triple in self.get_tuples(readable=False):
            subject = triple[0]
            predicate = triple[1]
            obj = triple[2]
            if is_uri(subject):
                uris.add(subject)
            if is_uri(predicate):
                uris.add(predicate)
            if is_uri(obj):
                uris.add(obj)
        return list(uris)
    
    def length(self) -> int:
        "The number of hops/tuples/triples in the path."
        return len(self.get_tuples())
    
    def get_verbalization(self) -> str:
        """
        Returns a verbalization of the path.
        
        e.g., "'A' has 'population' the '12345'"
        
        Returns:
            str: Verbalization of the path.
        """
        verbalization = ""
        variable_count = 0
        mapping = {}
        for tuple in self.get_tuples(readable=True):
            if tuple[0].startswith("?"):
                if tuple[0] not in mapping:
                    mapping[tuple[0]] = f"Entity_{variable_count}"
                    subject_label = f"Entity_{variable_count}"
                    variable_count += 1
                else:
                    subject_label = mapping[tuple[0]]
            else:
                subject_label = tuple[0]
            predicate_label = tuple[1]
            if tuple[2].startswith("?"):
                if tuple[2] not in mapping:
                    mapping[tuple[2]] = f"Entity_{variable_count}"
                    object_label = f"Entity_{variable_count}"
                    variable_count += 1
                else:
                    object_label = mapping[tuple[2]]
            else:
                object_label = tuple[2]
            verbalization += f"{subject_label} has {predicate_label} the {object_label}. "
        return verbalization.strip()
    
    def find_sample_values(self, k: int = 3):
        if len(self.sample_values) > 0:
            print("Sample values already exist, skipping SPARQL query.")
            return
        values = self.kg.get_values_for_triples(self.get_triples_string(readable=False), k)
        self.sample_values = values