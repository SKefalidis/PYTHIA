import argparse
import os
import json
from typing import Dict, List

from kgqa_toolkit.datasets import *

parser = argparse.ArgumentParser(description="Add topic entities to dataset")
parser.add_argument('--dataset_name', type=str, choices=DatasetFactory.list_datasets() + ['custom'], required=True, help="Name of the dataset")
parser.add_argument('--input_file', type=str, required=True, help="Path to the input dataset file")
parser.add_argument('--output_file', type=str, required=True, help="Path to the output dataset file")
args = parser.parse_args()

if not os.path.isfile(args.input_file):
    raise FileNotFoundError(f"Input file {args.input_file} does not exist")

new_dataset = []
empty_entries = 0
if args.dataset_name != 'custom':
    dataset = DatasetFactory.create_dataset(args.dataset_name, args.input_file)
    for entry in dataset.entries:
        question = entry.question
        query = entry.query
        topic_entities: Dict[str, str] = {}
        if 'topic_entities' in entry.raw_data:
            topic_entities: Dict[str, str] = entry.raw_data['topic_entities']
        elif 'topic_entity' in entry.raw_data:
            topic_entities: Dict[str, str] = entry.raw_data['topic_entity']
        else:
            raise ValueError("All datasets must have 'topic_entities' or 'topic_entity' field.")
        topic_entities_string = "" + ', '.join([f"{name} ({uri})" for name, uri in topic_entities.items()])

        new_dataset.append({
            'question': question,
            'query': query,
            'topic_entities': topic_entities_string
        })
else:
    with open(args.input_file, 'r') as f:
        dataset = json.load(f)
    for entry in dataset:
        question = entry.get('question', entry.get('Question', ''))
        # print(question)
        query = entry.get('query', entry.get('Sparql', ''))
        # print(query)
        topic_entities: Dict[str, str] = entry.get('topic_entities', {})
        if not topic_entities:
            empty_entries += 1
        #     raise ValueError("All datasets must have 'topic_entities' or 'topic_entity' field.")
        topic_entities_string = "" + ', '.join([f"{name} ({uri})" for name, uri in topic_entities.items()])

        new_dataset.append({
            'question': question,
            'query': query,
            'topic_entities': topic_entities_string
        })

with open(args.output_file, 'w') as f:
    json.dump(new_dataset, f, indent=4)

print(f"Added topic entities and saved to {args.output_file}")
print(f"Total entries with no topic entities: {empty_entries} out of {len(dataset)}")