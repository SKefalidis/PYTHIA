from typing_extensions import override, Optional
from src.engine.config import CONFIG
from src.knowledge_graphs.knowledge_graph import KnowledgeGraph
from src.utils import execute_sparql_query


class CustomKg(KnowledgeGraph):
    def __init__(self):
        super().__init__()
        
    @property
    def prefixes(self):
        return """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX uom: <http://www.opengis.net/def/uom/OGC/1.0/>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        """
    
    @override
    def register_uri_to_uril_mapping(self, uri: str):
        label_uris = CONFIG().get("kg_config")["label_uris"]
        if label_uris is not None:
            for label_uri in label_uris:
                # Support multi-hop label paths separated by '->'. Each element is a predicate.
                label = self.get_label(uri, label_uri)
                if label is not None:
                    label = label.replace(" ", "_")
                    self.uri_to_uril_map[uri] = uri + "_" + label
                    self.uril_to_uri_map[uri + "_" + label] = uri
                    return
        # Found no label, register the URI as is
        self.uri_to_uril_map[uri] = uri
        self.uril_to_uri_map[uri] = uri
            
    def get_label(self, uri: str, label_path_spec: str) -> Optional[str]:
        predicates = [p.strip() for p in label_path_spec.split('->') if p.strip()]
        if not predicates:
            return None

        # Build SPARQL patterns
        # If single predicate: <uri> p ?tailEntity .
        # If multiple: chain intermediate variables.
        def build_where(lang_filter: bool) -> str:
            where_clauses = []
            if len(predicates) == 1:
                where_clauses.append(f"<{uri}> <{predicates[0]}> ?tailEntity .")
            else:
                # Generate variable names ?v1 .. ?v{n-2}; final var is ?tailEntity
                var_names = [f"?v{i}" for i in range(1, len(predicates))]  # length-1 vars; last is tailEntity
                # Subject for first triple
                subj = f"<{uri}>"
                for idx, pred in enumerate(predicates):
                    is_last = (idx == len(predicates) - 1)
                    if is_last:
                        where_clauses.append(f"{subj} <{pred}> ?tailEntity .")
                    else:
                        obj_var = var_names[idx]
                        where_clauses.append(f"{subj} <{pred}> {obj_var} .")
                        subj = obj_var
            if lang_filter:
                # Treat empty language or English as acceptable (mirrors extractor logic)
                where_clauses.append("FILTER (lang(?tailEntity) = \"en\" || lang(?tailEntity) = \"\")")
            return "\n                ".join(where_clauses)

        # Try English/empty language first
        english_query = f"""
        SELECT ?tailEntity WHERE {{
                {build_where(lang_filter=True)}
        }} LIMIT 1
        """
        fallback_query = f"""
        SELECT ?tailEntity WHERE {{
                {build_where(lang_filter=False)}
        }} LIMIT 1
        """

        try:
            for q in (english_query, fallback_query):
                results = execute_sparql_query(q, self.endpoint).convert()
                bindings = results.get("results", {}).get("bindings", [])
                if bindings:
                    # print(bindings)
                    # exit(0)
                    return bindings[0]["tailEntity"]["value"]
        except Exception as e:
            print(f"Label path query error for {uri} via {label_path_spec}: {e}")
            return None
        return None