import math
from enum import Enum
from re import sub
import re
from typing import List, Tuple

from src.engine.qa.kg_explorer.kg_path import KgPath
from src.knowledge_graphs.knowledge_graph import Direction, KnowledgeGraph
from src.utils import execute_sparql_query, is_uri, return_sparql_query_results


class TargetType(Enum):
    URI = "uri"
    LITERAL = "literal"
    PLACEHOLDER = "placeholder"

class GraphSearchPathPart:
    def __init__(self, predicate: str, direction: Direction):
        self.predicate = predicate
        self.direction = direction
        if isinstance(direction, str):
            if direction == "outgoing":
                self.direction = Direction.OUTGOING
            elif direction == "incoming":
                self.direction = Direction.INCOMING
            elif direction == "unknown":
                self.direction = Direction.UNKNOWN
            else:
                raise ValueError(f"Invalid direction: {direction}")
    def __repr__(self):
        if self.direction == Direction.OUTGOING:
            return f"GraphSearchPathPath(--{self.predicate}->)"
        elif self.direction == Direction.INCOMING:
            return f"GraphSearchPathPath(<-{self.predicate}--)"
        elif self.direction == Direction.UNKNOWN:
            return f"GraphSearchPathPath(<-{self.predicate}->)"
        
    def get_label(self, kg: KnowledgeGraph):
        kg_component = kg.get_kg_component(self.predicate)
        if kg_component is not None:
            return kg_component.label
        else:
            return self.predicate

class GraphSearchPath(KgPath):

    # -----------------------
    # ----- Constructor -----
    # -----------------------
    
    def __init__(self, kg, parts: List[GraphSearchPathPart] = [], popularity: int = 0, sample_values: List[str] = [], leads_to: TargetType = None, start_uri: str = None, end_uri: str = None):
        super().__init__(kg)
        
        self.parts: List[GraphSearchPathPart] = parts
        self.popularity: int = popularity
        self.sample_values: List[str] = sample_values
        
        self.leads_to: TargetType = leads_to
        self.start_uri: str = start_uri
        self.end_uri: str = end_uri

    @classmethod
    def from_tuples(cls, kg, tuples: List[tuple], popularity: int = 0, sample_values: List[str] = [], leads_to: TargetType = None, start_uri: str = None, end_uri: str = None):
        parts = []
        for triple in tuples:
            subject = triple[0]
            predicate = triple[1]
            obj = triple[2]
            if subject.startswith("?") and not obj.startswith("?"):
                direction = Direction.INCOMING
            elif not subject.startswith("?") and obj.startswith("?"):
                direction = Direction.OUTGOING
            else:
                raise ValueError("Cannot determine direction from triple with two variables or two URIs.")
            part = GraphSearchPathPart(predicate, direction)
            parts.append(part)
        return cls(kg=kg, parts=parts, popularity=popularity, sample_values=sample_values, leads_to=leads_to, start_uri=start_uri, end_uri=end_uri)

    # --------------------------------------------
    # ----- Interface Methods Implementation -----
    # --------------------------------------------
        
    def get_tuples(self, readable: bool = False) -> List[Tuple[str, str, str]]:
        for part in self.parts:
            if part.direction == Direction.UNKNOWN:
                raise ValueError("Cannot generate triples for path with unknown direction parts.")
        # Generate triples
        tuples = []
        if not readable:
            current = self.start_uri # "<" + start + ">"
        else:
            label = self.kg.get_label(self.start_uri)
            current = label if label is not None else self.start_uri
        var_index = 0
        for idx, part in enumerate(self.parts):
            if idx == len(self.parts) - 1 and self.end_uri is not None:
                if not readable:
                    new_var = self.end_uri # "<" + goal + ">"
                else:
                    label = self.kg.get_label(self.end_uri)
                    new_var = label if label is not None else self.end_uri
            else:
                new_var = "?x" + str(var_index)
                var_index += 1

            if part.direction == Direction.OUTGOING:
                new_triple = (current, part.predicate, new_var)
            elif part.direction == Direction.INCOMING:
                new_triple = (new_var, part.predicate, current)
            else:
                raise ValueError("Cannot generate triples for path with unknown direction parts.")
            tuples.append(new_triple)
            current = new_var
        return tuples
    
    def get_cypher_string(self, readable: bool = False) -> str:
        print("WARNING: get_cypher_string returns normalized string.")
        return self._get_path_string_cypher_llm_friendly(readable=readable)
    
    def get_cypher_normalized_string(self, readable: bool = False) -> str:
        return self._get_path_string_cypher_llm_friendly(readable=readable)

    def get_predicates(self) -> List[str]:
        return [part.predicate for part in self.parts]
    
    def get_predicate_labels(self) -> List[str]:
        labels = []
        for part in self.parts:
            kg_component = self.kg.get_kg_component(part.predicate)
            if kg_component is not None:
                labels.append(kg_component.label)
            else:
                labels.append("Unnamed Predicate")
        return labels
    
    def get_predicate_directions(self) -> List[Direction]:
        return [part.direction for part in self.parts]
    
    # ------------------------------
    # ----- Additional Methods -----
    # ------------------------------
        
    def get_single_line_triples_string(self, readable: bool = False):
        return self.get_triples_string(separator=" ", readable=readable)
    
    def get_multi_line_triples_string(self, readable: bool = False):
        return self.get_triples_string(separator="\n", readable=readable)
    
    def get_path_triples_description(self, readable: bool = False, show_length: bool = True, show_specificity: bool = False):
        length_str = f"length: {self.length()}; " if show_length else ""
        specificity_str = f"specificity: {self.get_specificity():.2f}; " if show_specificity else ""
        return f"[{length_str}{specificity_str}matches: {self.popularity}] {self.get_single_line_triples_string(readable=readable)}"
    
    def get_path_string(self, readable: bool = False, llm_friendly: bool = False) -> str:
        if llm_friendly:
            return self._get_path_string_cypher_llm_friendly(readable=readable)
        else:
            return self._get_path_string_ad_hoc(readable=readable)
    
    def _get_path_string_ad_hoc(self, readable: bool = False):
        if readable and self.kg is None:
            raise ValueError("If readable is True, a KnowledgeGraph instance must be provided.")
        string = ""
        var_idx = 0
        current = self.kg.get_kg_component(self.start_uri).label if readable else self.start_uri
        for idx, part in enumerate(self.parts):
            string += current
            predicate_str = part.get_label(self.kg) if readable else part.predicate
            if part.direction == Direction.OUTGOING:
                string += f" -- {predicate_str} --> "
            elif part.direction == Direction.INCOMING:
                string += f" <-- {predicate_str} -- "
            elif part.direction == Direction.UNKNOWN:
                string += f" <-- {predicate_str} --> "
            current = f"?x{var_idx}"
            var_idx += 1
        string += self.kg.get_kg_component(self.end_uri).label if readable else (self.end_uri if self.end_uri is not None else "Unknown")
        return string
    
    def _get_path_string_cypher_original(self, readable: bool = False) -> str:
        return self._get_path_string_cypher(readable=readable, llm_friendly=False)
    
    def _get_path_string_cypher_llm_friendly(self, readable: bool = False) -> str:
        return self._get_path_string_cypher(readable=readable, llm_friendly=True)
    
    def _get_path_string_cypher(self, readable: bool = False, llm_friendly: bool = False) -> str:
        if readable and self.kg is None:
            raise ValueError("If readable is True, a KnowledgeGraph instance must be provided.")

        def to_node(val):
            return f"({val})"
        
        # 1. Initialize the start node
        start_val = self.kg.get_kg_component(self.start_uri).label if readable else self.start_uri
        curr_node_str = to_node(start_val)
        
        # We will build two versions simultaneously, then return the one requested
        linear_parts = [curr_node_str] # For standard Cypher: (A)<-[]-(B)
        triples = []                   # For LLMs: 1. (B)->(A)
        
        var_idx = 0

        for idx, part in enumerate(self.parts):
            predicate_str = part.get_label(self.kg) if readable else part.predicate
            
            # 2. Determine the 'next' node in the chain
            if idx < len(self.parts) - 1:
                next_val = f"?x{var_idx}"
                var_idx += 1
            else:
                next_val = self.kg.get_kg_component(self.end_uri).label if readable else self.end_uri
            
            next_node_str = to_node(next_val)

            # --- LOGIC A: Standard Linear Cypher (Your original logic) ---
            if not llm_friendly:
                if part.direction == Direction.OUTGOING:
                    arrow = f"-[:{predicate_str}]->"
                elif part.direction == Direction.INCOMING:
                    arrow = f"<-[:{predicate_str}]-"
                else: # UNKNOWN
                    arrow = f"<-[:{predicate_str}]->"
                
                linear_parts.append(arrow)
                linear_parts.append(next_node_str)

            # --- LOGIC B: LLM Friendly (Normalized SVO) ---
            else:
                # We enforce Subject -> Predicate -> Object ordering here
                if part.direction == Direction.INCOMING:
                    # Original: (Curr) <- (Next)
                    # Fixed:    (Next) -> (Curr)
                    step = f"{next_node_str}-[:{predicate_str}]->{curr_node_str}"
                
                elif part.direction == Direction.OUTGOING:
                    # Original: (Curr) -> (Next)
                    # Fixed:    (Curr) -> (Next)
                    step = f"{curr_node_str}-[:{predicate_str}]->{next_node_str}"
                
                else: # UNKNOWN
                    step = f"{curr_node_str}-[:{predicate_str}]-{next_node_str}"
                
                triples.append(step)

            # Advance the current node for the next iteration
            curr_node_str = next_node_str

        # 3. Return the requested format
        if llm_friendly:
            # Returns a clean list of steps separated by commas or newlines
            return " AND ".join(triples)
        
        return "".join(linear_parts)
    
    def get_path_description(self, readable: bool = False, llm_friendly: bool = False, show_length: bool = True, show_specificity: bool = True, with_sample_values: bool = False) -> str:
        length_str = f"length: {self.length()}; " if show_length else ""
        specificity_str = f"specificity: {self.get_specificity():.2f}; " if show_specificity else ""
        if self.popularity > 1 or not with_sample_values:
            return f"[{length_str}{specificity_str}matches: {self.popularity}] {self.get_path_string(readable=readable, llm_friendly=llm_friendly)}"
        else:
            string = f"[{length_str}{specificity_str}matches: {self.popularity}] {self.get_path_string(readable=readable, llm_friendly=llm_friendly)}"
            if self.sample_values is None or len(self.sample_values) == 0:
                # Becase the code is bad right now, we raise an error here to avoid silent failures and to purposefully find sample values.
                raise ValueError("No sample values available to include in path description.")
            for entry in self.sample_values:
                for var in entry.keys():
                    value = entry[var]
                    if is_uri(value):
                        kgc = self.kg.get_kg_component(value)
                        if kgc is not None:
                            label = self.kg.get_kg_component(value).label
                        else:
                            label = value
                    else:
                        label = value
                    string = string.replace("?" + var, str(label))
            return string
    
    def verbalize_path(self, kg: KnowledgeGraph, alternative: bool = False) -> str:
        verbalization = ""
        variable_count = 0
        mapping = {}
        for tuple in self.get_tuples():
            if tuple[0].startswith("?"):
                if tuple[0] not in mapping:
                    mapping[tuple[0]] = f"Entity_{variable_count}"
                    subject_label = f"Entity_{variable_count}"
                    variable_count += 1
                else:
                    subject_label = mapping[tuple[0]]
            else:
                subject_label = kg.get_kg_component(tuple[0]).label
            predicate_label = kg.get_kg_component(tuple[1]).label
            if tuple[2].startswith("?"):
                if tuple[2] not in mapping:
                    mapping[tuple[2]] = f"Entity_{variable_count}"
                    object_label = f"Entity_{variable_count}"
                    variable_count += 1
                else:
                    object_label = mapping[tuple[2]]
            else:
                object_label = kg.get_kg_component(tuple[2]).label
            if not alternative:
                verbalization += f"{subject_label} has {predicate_label} the {object_label}. "
            else:
                verbalization += f"'{subject_label}' is connected to '{object_label}' via '{predicate_label}'. "
        return verbalization.strip()
    
    # -------------------------------
    # ----- Getters and Setters -----
    # -------------------------------
    
    def push_part_front(self, part: GraphSearchPathPart):
        self.parts.insert(0, part)
    
    def add_part(self, part: GraphSearchPathPart):
        self.parts.append(part)

    def add_parts(self, parts: List[GraphSearchPathPart]):
        self.parts.extend(parts)
        
    def get_parts(self):
        return self.parts
        
    def add_sample_values(self, values):
        self.sample_values.extend(values)

    def get_sample_values(self):
        return self.sample_values

    def set_popularity(self, popularity: int):
        self.popularity = popularity

    def get_popularity(self):
        return self.popularity

    def set_leads_to(self, leads_to: TargetType):
        self.leads_to = leads_to
        
    def is_empty(self):
        return len(self.parts) == 0

    def length(self):
        return len(self.parts)
    
    def get_specificity(self):
        # 1. Setup constants
        # Total edges in KG are needed to calculate probability. 
        # If unavailable, use a large constant estimate or the max popularity seen.
        TOTAL_EDGES = 1_000_000_000
        
        # "Stop" predicates: important but high frequency.
        STRUCTURAL_PREDICATES = {'rdf:type', 'wdt:P31', 'rdfs:subClassOf'}
        
        specificity_score = 0.0
        
        for part in self.parts:
            predicate = part.predicate
            component = self.kg.get_kg_component(predicate)
            
            # Get raw frequency (ensure at least 1 to avoid log(0))
            # Combining incoming/outgoing is fine, or just use total edge instances.
            pred_freq = (component.incoming_edges_count + component.outgoing_edges_count)
            pred_freq = max(pred_freq, 1)
            
            # 2. Calculate Information Content (IC)
            # Probability P(p) = Frequency / Total
            p_pred = pred_freq / TOTAL_EDGES
            
            # IC = -log(P(p)). Result is usually between 0 (occurs everywhere) and 15+ (rare).
            # We add this to the score because Specificity is GOOD.
            ic = -math.log(p_pred)
            
            # 3. Handle Structural Predicates
            # if predicate in STRUCTURAL_PREDICATES:
                # OPTION A: Fixed Boost
                # Assign a "middle-ground" specificity so they aren't penalized like pure noise.
                # E.g., pretend they are reasonably rare.
                # ic = 5.0  
                
                # OPTION B: Scaling Factor
                # Multiply the low IC by a factor to make it 'important'
                # ic = ic * 2.0 

            specificity_score += ic

        # 4. Incorporate Path Popularity (The "Evidence" Score)
        # We also log-normalize this so it matches the scale of the IC score.
        # A path with 10,000 matches shouldn't simply overpower the semantic meaning.
        path_evidence = 0.0
        if self.popularity > 0:
            path_evidence = math.log(self.popularity)
        
        # 5. Final Calculation
        # We balance Semantic Specificity vs. Observational Evidence.
        # You can tune 'alpha' to weigh how much you care about the path being common.
        alpha = 1.0 
        
        final_score = specificity_score + (alpha * path_evidence)
        
        return final_score
        
    # -------------------------------------------
    # ----- Internal Representation Methods -----
    # -------------------------------------------
    
    def __str__(self):
        return self.get_path_string()

    def __repr__(self):
        return f"GraphSearchPath(parts={self.parts}, popularity={self.popularity}, start_uri={self.start_uri}, end_uri={self.end_uri})"

    def __eq__(self, other):
        if len(self.parts) != len(other.parts):
            return False
        for i in range(len(self.parts)):
            if self.parts[i].predicate != other.parts[i].predicate or self.parts[i].direction != other.parts[i].direction:
                return False
        return True