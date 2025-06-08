import os
import inspect
import time
import faiss
import numpy as np
import argparse
import sys
import jellyfish
import Levenshtein
from typing import List
from enum import Enum
from SPARQLWrapper import SPARQLWrapper, JSON, CSV, POST
from src.metrics import get_kgaqa_tracker
from sklearn.metrics.pairwise import cosine_similarity
from src.engine.config import CONFIG

from litellm import completion, drop_params
from litellm.types.utils import Usage, ModelResponse
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.exceptions import BadRequestError


# -----------------------------
# ----- General utilities -----
# -----------------------------

def get_relative_path(relative_path, stack_level=1):
    """
    Get the relative path of the current script.
    """
    caller_frame = inspect.stack()[stack_level]
    caller_filepath = caller_frame.filename

    caller_dir = os.path.dirname(os.path.abspath(caller_filepath))
    data_path = os.path.join(caller_dir, relative_path)
    return data_path

def read_configuration(config_path):
    import yaml
    with open(get_relative_path(config_path, 2), "r") as f:
        config = yaml.safe_load(f)
    return config

from src.logging import log, LogComponent, LogLevel, LogType # get_relative_path is used by logging.py

# -------------------------
# ----- LLM utilities -----
# -------------------------

def llm_call(llm: str, prompt: str|List, max_tokens: int = 500, temperature: float = 0.0, return_usage: bool = False):
    """
    Call the LLM with the given prompt and additional arguments.
    """
    get_kgaqa_tracker()._llm_calls += 1
    start_time = time.time()
    while True:
        try:
            messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            if "gpt-5" in llm:
                max_tokens = max_tokens * 4
            response: ModelResponse | CustomStreamWrapper = completion(model=llm, messages=messages, max_tokens=max_tokens, temperature=temperature, drop_params=True)
            if isinstance(response, CustomStreamWrapper):
                raise Exception("Streaming responses are not supported in `llm_call`.")
            usage: Usage = response.usage # type: ignore
            get_kgaqa_tracker()._llm_time += time.time() - start_time
            get_kgaqa_tracker()._llm_inputs += usage.prompt_tokens
            get_kgaqa_tracker()._llm_outputs += usage.completion_tokens
            get_kgaqa_tracker()._llm_tokens += usage.total_tokens
            generated = response['choices'][0]['message']['content']
            if not return_usage:
                return generated   
            else:
                return generated, usage
        except BadRequestError as e:
            print(f"Caught BadRequestError: {e}")
            # Logic to handle the limit (e.g., retrying with higher max_tokens)
            if "max_tokens" in str(e):
                print("The output limit was reached. Increasing tokens...")
                max_tokens *= 2
            else:
                raise e
        except Exception as e:
            get_kgaqa_tracker()._llm_time += time.time() - start_time
            print(f"Error calling LLM: {e} with {llm}")
            print(f"Prompt: {prompt}")
            return None
    
# ----------------------
# ----- Embeddings -----
# ----------------------

# # Use for BELA environment!   
# transformers_version = transformers.__version__
# if version.parse(transformers_version) >= version.parse("4.50.0") and embed_model is None:
#     from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#     embed_model = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True,
#                                     query_instruction="search_query: ",
#                                     text_instruction="search_document: ")

class TrackingEmbeddingWrapper:
    def __init__(self, model):
        self._model = model

    def get_query_embedding(self, text: str):
        tracker = get_kgaqa_tracker()
        tracker._embed_calls += 1
        start_time = time.time()
        embedding = self._model.get_query_embedding(text)
        tracker._embed_time += time.time() - start_time
        return embedding

    def get_text_embedding(self, text: str):
        tracker = get_kgaqa_tracker()
        tracker._embed_calls += 1
        start_time = time.time()
        embedding = self._model.get_text_embedding(text)
        tracker._embed_time += time.time() - start_time
        return embedding

    def __getattr__(self, name):
        """Forward everything else to the underlying model"""
        return getattr(self._model, name)

    def __dir__(self):
        """So dir() and autocomplete still show model attributes"""
        return list(set(dir(self._model) + list(self.__dict__.keys())))

_EMBED_MODEL = None

def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        raw_model = HuggingFaceEmbedding(
            model_name="nomic-ai/nomic-embed-text-v2-moe",
            trust_remote_code=True,
            query_instruction="search_query: ",
            text_instruction="search_document: ",
            embed_batch_size=64
        )
        _EMBED_MODEL = raw_model#TrackingEmbeddingWrapper(raw_model)
    return _EMBED_MODEL

def embed(text: str | List[str], is_query: bool = True):    
    get_kgaqa_tracker()._embed_calls += 1
    start_time = time.time()
    get_embed_model()._get_text_embedding
    if is_query:
        embedding = get_embed_model().get_query_embedding(text)
    else:
        if isinstance(text, list):
            embedding = get_embed_model().get_text_embedding_batch(text)
        else:
            embedding = get_embed_model().get_text_embedding(text)
    get_kgaqa_tracker()._embed_time += time.time() - start_time
    return embedding
    
# ----------------------------
# ----- SPARQL Execution -----
# ----------------------------

import time
from SPARQLWrapper import SPARQLWrapper, JSON
import socket
from urllib.error import URLError, HTTPError

class AuthorisedSPARQLWrapper(SPARQLWrapper):
    def __init__(self, endpoint_url):
        """
        Initialize the SPARQLWrapper with the endpoint URL.
        """
        super().__init__(endpoint_url)
        self.setCredentials(CONFIG().get("endpoint_username"), CONFIG().get("endpoint_password"))

def is_server_up(endpoint, test_query="ASK {}"):
    """
    Sends a lightweight test query to check if the SPARQL endpoint is responsive.
    Returns True if the server responds, False otherwise.
    """
    test_sparql = AuthorisedSPARQLWrapper(endpoint)
    test_sparql.setReturnFormat(JSON)
    test_sparql.setQuery(test_query)
    test_sparql.setMethod(POST)
    test_sparql.setTimeout(10)  # quick timeout
    try:
        test_sparql.query().convert()
        return True
    except Exception:
        return False

def execute_sparql_query(query, endpoint, max_wait_minutes=3, retry_interval=30, return_format=JSON):
    # print(query)
    get_kgaqa_tracker()._sparql_execs += 1
    start = time.time()

    if endpoint is None or endpoint == "PLACEHOLDER" or endpoint.strip() == "":
        print(query)
        print("ERROR: No endpoint URL provided. Please provide a valid SPARQL endpoint URL.", file=sys.stderr)
        sys.exit(1)

    sparql = AuthorisedSPARQLWrapper(endpoint)
    sparql.setReturnFormat(return_format)
    sparql.setQuery(query)
    sparql.setMethod(POST)
    sparql.setTimeout(max_wait_minutes * 60)

    exception = None
    try:
        query_result = sparql.query()
        get_kgaqa_tracker()._sparql_time += time.time() - start
        return query_result
    except (HTTPError, URLError, socket.timeout, socket.error) as e:
        exception = e
        print(f"[ERROR] SPARQL query error: {e}", file=sys.stderr)
        get_kgaqa_tracker()._sparql_time += time.time() - start
        log(f"[WARN] Query failed or timed out: {e}", "UTILITY")
        while not is_server_up(endpoint):
            print("\a")  # beep
            log(f"[WAIT] Server {endpoint} still down... retrying in {retry_interval} seconds", "UTILITY")
            time.sleep(retry_interval)
        
        sparql.setQuery("SELECT ?x { ?s ?p ?o } LIMIT 1")  # lightweight test query
        query_result = sparql.query()
        return query_result
    except Exception as e:
        exception = e
        get_kgaqa_tracker()._sparql_time += time.time() - start
        log(f"[ERROR] Unexpected failure: {e}", "UTILITY")

    # Retry logic: poll the server every `retry_interval` until it becomes responsive
    log("[INFO] Checking for server recovery...", "UTILITY")
    while not is_server_up(endpoint):
        print("\a")  # beep
        log(f"[WAIT] Server {endpoint} still down... retrying in {retry_interval} seconds", "UTILITY")
        time.sleep(retry_interval)

    raise exception  # Re-raise the last exception after retries

import csv
import io

def return_sparql_query_results(sparql_query, endpoint, return_error=False):
    try:
        is_ask = sparql_query.strip().upper().startswith("ASK")
        if is_ask:
            return_format = JSON  # ASK queries should return JSON for boolean results
        else:
            return_format = CSV  # SELECT queries can return CSV for easier parsing
        if return_format == JSON:
            if is_ask:
                query_result = execute_sparql_query(sparql_query, endpoint, return_format=return_format)
                results = query_result.convert()
                return [[str(results['boolean'])]]
        elif return_format == CSV:
            query_result = execute_sparql_query(sparql_query, endpoint, return_format=return_format)
            results = query_result.convert()
            result_str = (results.decode() if isinstance(results, bytes) else results)
            rows = list(csv.reader(io.StringIO(result_str)))
            return rows
    except Exception as e:
        log(f"Error get_types_for_node: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
        log(f"Query: {sparql_query}", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
        if return_error:
            return str(e)
        return []

def has_results(sparql_query, endpoint):
    results = return_sparql_query_results(sparql_query, endpoint)

    # ASK query: results look like [[True]] or [[False]]
    if sparql_query.strip().upper().startswith("ASK"):
        if results and results[0][0].lower() == 'true':
            return True
        return False

    # SELECT query: results contain a header row + data rows
    if results and len(results) > 1:
        return True
    return False
    
def endpoints_fill_parse_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup: 
    endpoint_group = parser.add_argument_group("Endpoints")
    endpoint_group.add_argument("--endpoint_server", type=str, required=False,
                                help="Which GraphDB server to use")
    endpoint_group.add_argument("--endpoint_server_manager", type=str, required=False,
                                help="Which GraphDB Manager endpoint to use for repository management operations")
    endpoint_group.add_argument("--endpoint_username", type=str, required=False, default=None,
                                help="Username for the triplestore endpoint (if required)")
    endpoint_group.add_argument("--endpoint_password", type=str, required=False, default=None,
                                help="Password for the triplestore endpoint (if required)")
    return endpoint_group

# ---------------------------
# ----- FAISS utilities -----
# ---------------------------
    
def load_faiss_index(index_dir, nprobe=64):
    """
    Load the FAISS index and set the nprobe parameter for high-accuracy PQ search.
    """
    print("Loading FAISS index...")
    index = faiss.read_index(index_dir + "/faiss.index")

    # Set nprobe if it's an IVF/IVFPQ index
    if isinstance(index, faiss.IndexPreTransform):
        # OPQ wrapper: get the inner index
        inner_index = index.index
        if isinstance(inner_index, faiss.IndexIVFPQ):
            inner_index.nprobe = nprobe
            print(f"Set nprobe={nprobe} for inner IVFPQ index")
        else:
            print("Warning: inner index is not IVFPQ")
    elif isinstance(index, faiss.IndexIVFPQ) or isinstance(index, faiss.IndexIVF):
        index.nprobe = nprobe
        print(f"nprobe set to {nprobe}")
    else:
        print("Index is not IVFPQ, nprobe will be ignored")

    print("Loading documents...")
    with open(index_dir + '/docs.txt', "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines()]
        
    embedding_map = {}
    for i, doc in enumerate(documents):
        key = doc.split('\t')[0].strip()
        embedding_map[key] = i

    return index, documents, embedding_map

def search_faiss_index(index, documents, query, k=5, debug=False, threshold=0.0):
    query_vector = get_embed_model().get_query_embedding(query)
    query_vector = np.array(query_vector).astype("float32").reshape(1, -1)


    # Normalize query vector for cosine similarity if index is IVFPQ/OPQ or Flat (if built with normalized vectors)
    should_normalize = False
    if isinstance(index, faiss.IndexPreTransform):
        # OPQ wrapper: get the inner index
        inner_index = index.index
        if isinstance(inner_index, faiss.IndexIVFPQ):
            should_normalize = True
    elif isinstance(index, faiss.IndexIVFPQ):
        should_normalize = True
    elif isinstance(index, faiss.IndexFlatL2):
        # If your flat index was built with normalized vectors, enable this
        should_normalize = True

    if should_normalize:
        faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, k)
    
    results = []
    cosine_similarities = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        similarity = 1 - (dist / 2)  # convert L2 distance to cosine similarity
        results.append(documents[idx])
        if debug:
            print(f"[Distance: {dist:.4f}] {documents[idx]}")
        cosine_similarities.append(similarity)
    return results, cosine_similarities

# ---------------------
# ----- simstring -----
# ---------------------
import simstring
import glob

class SimStringShardedReader:
    """
    Wrapper to search across multiple SimString shard DBs loaded in memory.
    All shard readers remain open until explicitly closed.
    """
    def __init__(self, readers: list):
        # readers is a list of simstring.reader objects, already opened
        self._readers = readers
        self._measure = simstring.overlap
        self._threshold = 0.7

    @property
    def measure(self):
        return self._measure

    @measure.setter
    def measure(self, value):
        self._measure = value
        for r in self._readers:
            r.measure = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value
        for r in self._readers:
            r.threshold = value

    def retrieve(self, query: str):
        results = []
        for r in self._readers:
            results.extend(r.retrieve(query))
        return results

    def close(self):
        for r in self._readers:
            try:
                r.close()
            except Exception:
                pass


def load_simstring_index(index_dir):
    print("Loading SimString index...")
    import pickle
    db_path = os.path.join(index_dir, "keys.db")
    pkl_path = os.path.join(index_dir, "key_to_value.pkl")

    index_obj = None
    if os.path.exists(db_path):
        # Single DB
        index = simstring.reader(db_path)
        # index.measure = simstring.overlap
        # index.threshold = 0.7
        index_obj = index
    else:
        # Look for shards: index_dir/shard_*/keys.db
        shard_paths = sorted(glob.glob(os.path.join(index_dir, "shard_*", "keys.db")))
        if not shard_paths:
            raise FileNotFoundError(f"No SimString DB found at {db_path} or shards under {index_dir}")
        # Pre-open all shard readers and keep them in memory
        readers = []
        for path in shard_paths:
            r = simstring.reader(path)
            readers.append(r)
        shard_index = SimStringShardedReader(readers)
        shard_index.measure = simstring.overlap
        shard_index.threshold = 0.7
        index_obj = shard_index

    # Load key->value mapping saved alongside the DB(s)
    key_to_value = {}
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as pf:
                key_to_value = pickle.load(pf)
        except Exception as e:
            print(f"Warning: failed to load key_to_value mapping from {pkl_path}: {e}")
    else:
        print(f"Warning: mapping file not found at {pkl_path}; search will return keys")

    print("SimString index loaded.")
    return index_obj, key_to_value

class Similarity(Enum):
    JARO_WINKLER = 1
    LEVENSHTEIN = 2
    COSINE = 3

def search_simstring_index(index, key_to_value, query, similarity, k=5, threshold=0.0, debug=False, length_filter=True):
    """
    Search a simstring index and rank retrieved keys by Jaro-Winkler similarity.
    Returns the top-k most similar mapped values.
    """
    # Retrieve all candidate keys from the index
    # start = time.time()
    results_keys = list(index.retrieve(query))
    # print(f"SimString retrieved {len(results_keys)} candidates for query: {query}")
    # print(f"SimString retrieval took {time.time() - start:.4f} seconds")

    if not results_keys:
        return [], []
    
    query_length = len(query)
    
    if length_filter:
        results_keys = [key for key in results_keys if abs(len(key) - query_length) <= 5 and key.strip() != ""]  # remove empty keys
        if not results_keys:
            return [], []
    
    # start = time.time()
    if similarity == Similarity.COSINE:
        embeddings = embed(results_keys, is_query=False)
        query_embedding = embed(query, is_query=False)
    # print(f"Embedding computation took {time.time() - start:.4f} seconds")

    # Compute Jaro-Winkler similarity for each key
    # start = time.time()
    scored = []
    for idx, key in enumerate(results_keys):
        if similarity == Similarity.JARO_WINKLER:
            score = jellyfish.jaro_winkler_similarity(query, key)
        elif similarity == Similarity.LEVENSHTEIN:
            score = Levenshtein.ratio(query, key)
        elif similarity == Similarity.COSINE:
            score = cosine_similarity(
                np.array(embeddings[idx]).reshape(1, -1), # make into 2D array
                np.array(query_embedding).reshape(1, -1)
            )[0][0]
        scored.append((key, score))
        if debug:
            print(f"{key} -> {score:.4f}")

    # Sort by score (highest first)
    # print(f"Scored {len(scored)} candidates.")
    # for key, score in scored:
    #     entries_for_key = key_to_value.get(key, [])
    #     print(f"[Score: {score:.4f}] {key} -> {len(entries_for_key)} entries")
    scored.sort(key=lambda x: x[1], reverse=True)
    # print(f"Scoring took {time.time() - start:.4f} seconds")

    # Map to values using key_to_value dictionary
    # start = time.time()
    mapped = []
    scores = []
    for key, score in scored[:k]:
        # if score < threshold:
        #     continue
        value_list = key_to_value.get(key, [])
        for val in value_list:
            mapped.append(val+'\t'+key)
            scores.append(score)
            if debug:
                print(f"[Match] {val} ({key}) = {score:.4f}")
        if len(mapped) >= k:
            break
    # print(f"Mapping took {time.time() - start:.4f} seconds")

    # if k < 1:
    return mapped, scores
    # else:
    #     return mapped[:k], scores[:k]

# ---------------------------------
# ----- String type utilities -----
# ---------------------------------

def is_uri(s):
    if s is None or not isinstance(s, str):
        return False
    return "http://" in s

def is_entity_placeholder(s: str):
    return isinstance(s, str) and s.isupper()

def is_property_description(s: str):
    return not is_entity_placeholder(s)

def is_type_predicate(str):
    if "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" in str:
        return True
    if "http://www.wikidata.org/prop/direct/P31" in str:
        return True
    return False

# ---------------------------
# ----- Endpoints Setup -----
# ---------------------------

import requests

def setup_graphdb(endpoint: str):
    from yarl import URL
    server_manager_url = CONFIG().get("endpoint_server_manager")
    if server_manager_url is None:
        raise Exception("Endpoint server manager URL is not configured.")
    graphdb_server_restart_endpoint = URL(server_manager_url) / "clear"
    graphdb_server_load_endpoint = URL(server_manager_url) / "load"
    
    response = requests.post(graphdb_server_restart_endpoint)
    if response.status_code != 200:
        print(f"Failed to restart GraphDB server. Status code: {response.status_code}, Response: {response.text}")
        raise Exception("Failed to restart GraphDB server.")
        
    time.sleep(10)
    
    payload = {
        "endpoint": endpoint,
    }
    
    response = requests.post(graphdb_server_load_endpoint, json=payload)
    if response.status_code != 200:
        print(f"Failed to load repositories in GraphDB server. Status code: {response.status_code}, Response: {response.text}")
        raise Exception("Failed to load repositories in GraphDB server.")


def setup_graph_tool(graph_string: str) -> bool:    
    AVAILABLE_GRAPHS = ["dbpedia10", "dbpedia2016", "freebase", "wikidata"]
    
    endpoint = CONFIG().get("endpoint_graph_search_server") + "load-graph"
    
    if graph_string not in AVAILABLE_GRAPHS:
        print(f"Graph {graph_string} is not available. Available graphs: {AVAILABLE_GRAPHS}")
        exit(1)
            
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "source_dir": graph_string,
        "filter_classes": "1",
    }

    response = requests.post(endpoint, json=payload, headers=headers)
    if response.status_code != 200:
        print(f"Failed to load graph tool for graph {graph_string}. Status code: {response.status_code}, Response: {response.text}")
        exit(1)
