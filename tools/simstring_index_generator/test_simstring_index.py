import os
import glob
import argparse
import simstring
import jellyfish
import Levenshtein
from enum import Enum

class SimStringShardedReader:
    """Keeps all shard readers open in memory; provides unified retrieve()."""
    def __init__(self, readers):
        self._readers = readers
        self.measure = simstring.overlap
        self.threshold = 0.7
        # propagate defaults
        for r in self._readers:
            r.measure = self.measure
            r.threshold = self.threshold

    def retrieve(self, query):
        results = []
        for r in self._readers:
            results.extend(r.retrieve(query))
        return results

def load_simstring_index(index_dir):
    print("Loading SimString index...")
    import pickle
    db_path = os.path.join(index_dir, "keys.db")
    pkl_path = os.path.join(index_dir, "key_to_value.pkl")

    index = None
    if os.path.exists(db_path):
        # Single DB
        index = simstring.reader(db_path)
        index.measure = simstring.overlap
        index.threshold = 0.7
    else:
        # Sharded layout
        shard_paths = sorted(glob.glob(os.path.join(index_dir, "shard_*", "keys.db")))
        if not shard_paths:
            raise FileNotFoundError(f"No SimString DB found at {db_path} or shards under {index_dir}")
        readers = []
        for sp in shard_paths:
            r = simstring.reader(sp)
            readers.append(r)
        index = SimStringShardedReader(readers)
        print(f"Loaded {len(readers)} shards.")

    # Load key->value mapping saved alongside the DB
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
    return index, key_to_value

class Similarity(Enum):
    JARO_WINKLER = 1
    LEVENSHTEIN = 2

def search_simstring_index(index, key_to_value, query, similarity, k=5, threshold=0.0, debug=False):
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
        return []

    # Compute Jaro-Winkler similarity for each key
    # start = time.time()
    scored = []
    for idx, key in enumerate(results_keys):
        if similarity == Similarity.JARO_WINKLER:
            score = jellyfish.jaro_winkler_similarity(query, key)
        elif similarity == Similarity.LEVENSHTEIN:
            score = Levenshtein.ratio(query, key)
        scored.append((key, score))
        if debug:
            print(f"{key} -> {score:.4f}")

    # Sort by score (highest first)
    scored.sort(key=lambda x: x[1], reverse=True)
    # print(f"Scoring took {time.time() - start:.4f} seconds")

    # Map to values using key_to_value dictionary
    # start = time.time()
    mapped = []
    for key, score in scored[:k]:
        if score < threshold:
            continue
        value_list = key_to_value.get(key, [])
        for val in value_list:
            mapped.append(val+'\t'+key)
            if debug:
                print(f"[Match] {val} ({key}) = {score:.4f}")
    # print(f"Mapping took {time.time() - start:.4f} seconds")

    return mapped[:k]

if __name__ == "__main__":
    argparse.ArgumentParser(description="simstring Dense Index Playground")
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the index")
    args = parser.parse_args()
    
    index, documents = load_simstring_index(args.path)
    
    while True:
        user_input = input("Enter your query (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        else:
            print(f"User input: {user_input}")
            response = search_simstring_index(index, documents, user_input, similarity=Similarity.JARO_WINKLER)
            print(response)