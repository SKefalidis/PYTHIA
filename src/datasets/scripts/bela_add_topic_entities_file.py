"""
This script adds topic entities to the given dataset file.
"""

from src.engine.config import CONFIG
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs


if __name__ == '__main__':
    import argparse
    import os
    import json
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description="Add topic entities to dataset")
    parser.add_argument('--kg', type=str, choices=['wikidata', 'dbpedia10', 'dbpedia2016', 'freebase'], required=True, help="Name of the dataset")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input dataset file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output dataset file")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"Input file {args.input_file} does not exist")

    with open(args.input_file, 'r') as f:
        dataset = json.load(f)
    
    kg = None
    if args.kg == 'wikidata':
        kg = KnowledgeGraphs.WIKIDATA
    elif args.kg == 'dbpedia10':
        kg = KnowledgeGraphs.DBPEDIA10
    elif args.kg == 'dbpedia2016':
        kg = KnowledgeGraphs.DBPEDIA
    elif args.kg == 'freebase':
        kg = KnowledgeGraphs.FREEBASE
    kg.load(os.path.join(CONFIG().get("index_dir"), args.kg))
    
    entries = []
    for idx in range(1, len(dataset)):
        entry = dataset[idx]
        question = entry['question']
        predicted_entities = entry['predictions']
        predicted_entities_labels = [kg.get_label_from_uri(ent) for ent in predicted_entities if ent is not None]
        
        topic_entities = {}
        for uri, label in zip(predicted_entities, predicted_entities_labels):
            topic_entities[uri] = label
        
        new_entry = {
            'question': question,
            'topic_entity': topic_entities,
            'topic_entities': topic_entities
        }
        
        entries.append(new_entry)

    with open(args.output_file, 'w') as f:
        json.dump(entries, f, indent=4)

    print(f"Added topic entities and saved to {args.output_file}")
