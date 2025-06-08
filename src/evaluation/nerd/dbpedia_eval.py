# ---------------------------------------------------------------------
# ----- Useful for counting redirects in DBpedia NERD evaluation. -----
# ---------------------------------------------------------------------

from SPARQLWrapper import SPARQLWrapper, JSON
import json
from tqdm import tqdm
import argparse

# SPARQL endpoint for DBpedia
sparql = SPARQLWrapper("https://dbpedia.org/sparql")

# Cache to avoid repeated queries
redirect_cache = {}

def resolve(uri):
    if uri in redirect_cache:
        return redirect_cache[uri]

    query = f"""
    SELECT ?target WHERE {{
        <{uri}> <http://dbpedia.org/ontology/wikiPageRedirects> ?target
    }} LIMIT 1
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            target = bindings[0]["target"]["value"]
            redirect_cache[uri] = target
            return target
    except Exception as e:
        print(f"SPARQL query failed for {uri}: {e}")

    # No redirect found, return original
    redirect_cache[uri] = uri
    return uri


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process a list of datasets."
    )
    
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to the JSON file containing NERD predictions for a DBpedia dataset.")
    
    args = parser.parse_args()
    
    # --- Load your main data file ---
    file_path = args.predictions
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = data[1:] # ignore metrics

    # Metrics counters
    tp = fp = fn = 0

    for entry in tqdm(entries):
        pred_raw = entry.get('predictions', [])
        gold_raw = entry.get('gold', [])

        pred_resolved = {resolve(uri) for uri in pred_raw}
        gold_resolved = {resolve(uri) for uri in gold_raw}

        entry_tp = len(pred_resolved & gold_resolved)
        entry_fp = len(pred_resolved - gold_resolved)
        entry_fn = len(gold_resolved - pred_resolved)

        tp += entry_tp
        fp += entry_fp
        fn += entry_fn

    # Final metrics
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("Micro-averaged metrics (live redirect resolution):")
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Precision:       {precision:.4f}")
    print(f"  Recall:          {recall:.4f}")
    print(f"  F1-score:        {f1:.4f}")
