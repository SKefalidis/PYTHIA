"""
This script adds topic entities to the given dataset file.
"""

from src.dataloaders.dataset import DatasetFactory
from src.linking.class_linking.gold_class_linker import GoldClassLinker
from src.linking.entity_linking.gold_entity_linker import GoldEntityLinker
from src.config import CONFIG
from src.utils import endpoints_fill_parse_args


if __name__ == '__main__':
    import argparse
    import os
    import json
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description="Add topic entities to dataset")
    parser.add_argument('--index_name', type=str, required=True, help="Name of the knowledge graph index to load")
    parser.add_argument('--dataset_name', type=str, choices=DatasetFactory.list_datasets(), required=True, help="Name of the dataset")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input dataset file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output dataset file")

    endpoints_fill_parse_args(parser)

    args = parser.parse_args()

    # Initialize configuration with parser arguments
    CONFIG(args)

    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"Input file {args.input_file} does not exist")
    
    dataset = DatasetFactory.create_dataset(args.dataset_name, args.input_file)
    
    dataset.get_knowledge_graph().load(os.path.join(CONFIG().get("index_dir"), args.index_name))
    
    if args.dataset_name != "materials":
        entity_linker = GoldEntityLinker(dataset.get_knowledge_graph(), dataset.get_prefixes())
        class_identifier = GoldClassLinker(dataset.get_knowledge_graph(), endpoint_url=dataset.get_knowledge_graph().endpoint if args.endpoint_server is None else args.endpoint_server, prefixes=dataset.get_prefixes())
        
        empty_entries = 0
        for entry in tqdm(dataset):
            sparql = dataset.get_query(entry)
            classes = class_identifier.identify(sparql)
            classes = {c : dataset.get_knowledge_graph().get_label_from_uri(c) for c in classes}
            entities = entity_linker.identify(sparql)
            entities = {e : dataset.get_knowledge_graph().get_label_from_uri(e) for e in entities}
            if not classes and not entities:
                print(f"Warning: No topic entities found for question: {dataset.get_question(entry)}")
                empty_entries += 1
            entry['topic_entities'] = {**classes, **entities}

    with open(args.output_file, 'w') as f:
        json.dump(dataset.dataset, f, indent=4)

    print(f"Added topic entities and saved to {args.output_file}")
    print(f"Total entries with no topic entities: {empty_entries} out of {len(dataset)}")
