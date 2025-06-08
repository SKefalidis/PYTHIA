import re
import requests
import textwrap
from typing_extensions import override, Optional
from src.knowledge_graphs.knowledge_graph import KnowledgeGraph


class Wikidata(KnowledgeGraph):
    
    @property
    def prefixes(self):
        return textwrap.dedent("""
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX p: <http://www.wikidata.org/prop/>
            PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
            PREFIX pqn: <http://www.wikidata.org/prop/qualifier/value-normalized/>
            PREFIX pqv: <http://www.wikidata.org/prop/qualifier/value/>
            PREFIX pr: <http://www.wikidata.org/prop/reference/>
            PREFIX prn: <http://www.wikidata.org/prop/reference/value-normalized/>
            PREFIX prov: <http://www.w3.org/ns/prov#>
            PREFIX prv: <http://www.wikidata.org/prop/reference/value/>
            PREFIX ps: <http://www.wikidata.org/prop/statement/>
            PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/>
            PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX schema: <http://schema.org/>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdno: <http://www.wikidata.org/prop/novalue/>
            PREFIX wds: <http://www.wikidata.org/entity/statement/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wdtn: <http://www.wikidata.org/prop/direct-normalized/>
            PREFIX wdv: <http://www.wikidata.org/value/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        """)
        
    @property
    def entity_popularity_k(self) -> int:
        "Normally set to the median, we set it higher to bias towards popular entities."
        return 112
    
    @property
    def class_popularity_k(self) -> int:
        "Median popularity of classes."
        return 81
    
    @property
    def predicate_popularity_k(self) -> int:
        "Median popularity of predicates."
        return 889
    
    @override
    def register_uri_to_uril_mapping(self, uri: str):
        kg_component = self.get_kg_component(uri)
        if kg_component is not None:
            label = kg_component.label
        else: # Fallback: try to get label from Wikidata API
            label = self.get_wikidata_label(uri.split("/")[-1], lang="en")
        if label is not None:
            label = label.replace(" ", "_")
            self.uri_to_uril_map[uri] = uri + "_" + label
            self.uril_to_uri_map[(uri + "_" + label)] = uri
        else:
            self.uri_to_uril_map[uri] = uri
            self.uril_to_uri_map[uri] = uri
            
    def get_wikidata_label(self, qid, lang="en") -> Optional[str]:       
        if not isinstance(qid, str) or not re.fullmatch(r"[QP]\d+", qid):
            return None

        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "format": "json",
            "props": "labels",
            "languages": lang
        }
        headers = {
            "User-Agent": "QuestionAnsweringQidTranslator/1.0 (https://ai.di.uoa.gr/; skefalidis@di.uoa.gr)"
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=(5, 5))
            response.raise_for_status()
            data = response.json()
            return data["entities"][qid]["labels"][lang]["value"]
        except (KeyError, requests.exceptions.RequestException) as e:
            print(f"Error: {e}")
            return None