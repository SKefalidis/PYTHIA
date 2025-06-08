from typing import List
import requests
import requests
from urllib.parse import urlparse, unquote
from abc import abstractmethod

from src.evaluation.evaluatable import Evaluatable
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.utils import execute_sparql_query


class EntityLinker(Evaluatable):

    def __init__(self, knowledge_graph: KnowledgeGraphs):
        self.knowledge_graph = knowledge_graph
        self.convert = False

    @abstractmethod
    def nerd(self, question: str, debug: bool = False, logging: bool = False):
        pass
    
    def identify(self, question: str):
        return self.nerd(question)
    
    def predict(self, question: str, debug: bool = False, logging: bool = False):
        return self.nerd(question, debug=debug, logging=logging)
    
    def convert_to_kg(self, uris: List[str]):
        if self.knowledge_graph in self.supported_targets():
            return uris
        
        supported_to_wikipedia_func = None
        if KnowledgeGraphs.DBPEDIA in self.supported_targets() or KnowledgeGraphs.DBPEDIA10 in self.supported_targets():
            supported_to_wikipedia_func = dbpedia_to_wikipedia
        elif KnowledgeGraphs.WIKIDATA in self.supported_targets():
            supported_to_wikipedia_func = wikidata_to_wikipedia
        # elif KnowledgeGraphs.YAGO2geo in self.supported_targets():
        #     supported_to_wikipedia_func = yago2_to_wikipedia
        
        wikipedia_to_target_func = None
        if self.knowledge_graph == KnowledgeGraphs.DBPEDIA or self.knowledge_graph == KnowledgeGraphs.DBPEDIA10:
            wikipedia_to_target_func = wikipedia_to_dbpedia
        elif self.knowledge_graph == KnowledgeGraphs.WIKIDATA:
            wikipedia_to_target_func = wikipedia_to_wikidata
        # elif self.knowledge_graph == KnowledgeGraphs.YAGO2geo or self.knowledge_graph == KnowledgeGraphs.ELECTIONS_KG or self.knowledge_graph == KnowledgeGraphs.TERRAQ_KG:
        #     wikipedia_to_target_func = wikipedia_to_yago2
        elif self.knowledge_graph == KnowledgeGraphs.FREEBASE:
            if KnowledgeGraphs.WIKIDATA in self.supported_targets():
                supported_to_wikipedia_func = lambda x: x
                wikipedia_to_target_func = wikidata_to_freebase
            else:
                # original to wikipedia to wikidata to freebase
                if supported_to_wikipedia_func is not None:
                    orig_func = supported_to_wikipedia_func
                    supported_to_wikipedia_func = lambda x: wikipedia_to_wikidata(orig_func(x))
                wikipedia_to_target_func = wikidata_to_freebase
               
        if wikipedia_to_target_func is None: # did not find a valid conversion :-(
            print(f"WARNING: No valid conversions between {self.supported_targets()} and {self.knowledge_graph}")
            return uris
        
        converted_uris = []
        for uri in uris:
            # print(uri)
            wikipedia_url = supported_to_wikipedia_func(uri)
            # print(wikipedia_url)
            if wikipedia_url is None:
                continue
            new_uri = wikipedia_to_target_func(wikipedia_url)
            converted_uris.append(new_uri)
                
        return converted_uris
                
    
    # FIXME: Since it is not needed here, maybe it should not be in the parent class :^)
    def get_resource(self):
        return ""
    
    @abstractmethod
    def supported_targets(self) -> List[KnowledgeGraphs]:
        pass


def wikidata_to_wikipedia(wikidata_url, language='en'):
    return wikidata_id_to_wikipedia(wikidata_url.split('/')[-1])


def wikidata_id_to_wikipedia(wikidata_id, language='en'):
    """
    Convert a Wikidata ID to a Wikipedia article link.

    :param wikidata_id: The Wikidata ID (e.g., 'Q99')
    :param language: The language code for the Wikipedia (e.g., 'en' for English)
    :return: The Wikipedia article URL or None if not found
    """
    # Wikidata API endpoint
    url = f"https://www.wikidata.org/w/api.php"

    # Parameters for the API request
    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "format": "json",
        "props": "sitelinks"
    }
    headers = {
        "User-Agent": "WikidataEngine/1.0 (https://ai.di.uoa.gr/; skefalidis@di.uoa.gr)"
    }

    # Make the request to the Wikidata API
    response = requests.get(url, params=params, headers=headers)
    try:
        data = response.json()
    except Exception as e:
        print(f"Error parsing JSON response from Wikidata API: {e}")
        print(response.text)
        return None

    # Check if the Wikidata ID exists in the response
    if 'entities' in data and wikidata_id in data['entities']:
        entity = data['entities'][wikidata_id]
        sitelinks = entity.get("sitelinks", {})

        # Construct the Wikipedia URL based on the specified language
        lang_key = f"{language}wiki"
        if lang_key in sitelinks:
            title = sitelinks[lang_key]['title']
            wikipedia_url = f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}"
            return wikipedia_url

    # Return None if no Wikipedia article is found
    return None


def wikipedia_to_wikidata(wikipedia_url):
    # Extract the title from the Wikipedia URL
    parsed_url = urlparse(wikipedia_url)
    if "wikipedia.org" not in parsed_url.netloc:
        print(f"WARNING: Not a valid Wikipedia URL: {wikipedia_url}. Returning as is.")
        return wikipedia_url
        # raise ValueError("Not a valid Wikipedia URL")

    language = parsed_url.netloc.split('.')[0]
    title = unquote(parsed_url.path.split('/')[-1])

    # Query the Wikipedia API for page properties
    api_url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageprops",
        "format": "json"
    }
    headers = {
        "User-Agent": "WikidataEngine/1.0 (https://ai.di.uoa.gr/; skefalidis@di.uoa.gr)"
    }

    response = requests.get(api_url, params=params, headers=headers)
    data = response.json()

    pages = data["query"]["pages"]
    for page_id, page_data in pages.items():
        if "pageprops" in page_data and "wikibase_item" in page_data["pageprops"]:
            wikidata_id = page_data["pageprops"]["wikibase_item"]
            return f"http://www.wikidata.org/entity/{wikidata_id}"

    return None


def wikipedia_id_to_wikidata(pageid):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "pageids": pageid,
        "prop": "pageprops",
        "format": "json"
    }

    response = requests.get(url, params=params)
    data = response.json()

    try:
        page_info = data["query"]["pages"][str(pageid)]
        return page_info["pageprops"]["wikibase_item"]
    except KeyError:
        return None


def wikipedia_to_yago2(wikipedia_url):
    return wikipedia_url.replace('https://en.wikipedia.org/wiki/', 'http://yago-knowledge.org/resource/')


def yago2_to_wikipedia(yago_url):
    return yago_url.replace('http://yago-knowledge.org/resource/', 'https://en.wikipedia.org/wiki/')


def wikipedia_to_dbpedia(wikipedia_url):
    return wikipedia_url.replace('https://en.wikipedia.org/wiki/', 'http://dbpedia.org/resource/')


def dbpedia_to_wikipedia(dbpedia_url):
    return dbpedia_url.replace('http://dbpedia.org/resource/', 'https://en.wikipedia.org/wiki/')


def wikidata_to_freebase(wikidata_uri):
    query = f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT ?freebaseID WHERE {{
      <{wikidata_uri}> wdt:P646 ?freebaseID .
    }}
    """
    
    try:
        results = execute_sparql_query(query, KnowledgeGraphs.WIKIDATA.endpoint).convert()
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            return "http://rdf.freebase.com/ns/" + bindings[0]["freebaseID"]["value"][1:].replace("/", ".")
        else:
            return None
    except Exception as e:
        print(f"Error querying SPARQL: {e}")
        return None

if __name__ == '__main__':
    # Example usage
    wikidata_id = "http://www.wikidata.org/entity/Q42"  # Douglas Adams
    freebase_id = wikidata_to_freebase(wikidata_id)

    if freebase_id:
        print(f"Freebase ID for {wikidata_id}: {freebase_id}")
    else:
        print(f"No Freebase ID found for {wikidata_id}")
