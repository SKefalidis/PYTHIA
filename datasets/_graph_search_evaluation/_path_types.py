import json
from typing import Dict


def calculate_path_types(dataset: Dict):
    path_type_counts = {
        "named_to_named": 0,
        "named_to_class": 0,
        "class_to_named": 0,
        "class_to_class": 0,
        "unknown": 0
    }
    
    for sample in dataset:
        named_entities = sample.get("named_entities", [])
        classes = sample.get("classes", [])
        paths = sample.get("known_to_known_paths", [])
        
        for path in paths:
            start_uri = path[0]
            end_uri = path[-1]
            
            if start_uri in named_entities and end_uri in named_entities:
                path_type = "named_to_named"
            elif start_uri in named_entities and end_uri in classes:
                path_type = "named_to_class"
            elif start_uri in classes and end_uri in named_entities:
                path_type = "class_to_named"
            elif start_uri in classes and end_uri in classes:
                path_type = "class_to_class"
                print(f"Class to class path found: {path}")
            else:
                path_type = "unknown"
                print(f"Unknown path type for start: {start_uri}, end: {end_uri}")
            path_type_counts[path_type] += 1
    return path_type_counts

dataset_names_paths = [
    ("Beastiary_graph_exploration_evaluation.jsonl", "beastiary"),
    ("WebQSP_graph_exploration_evaluation.jsonl", "webqsp"),
    ("CWQ_graph_exploration_evaluation.jsonl", "cwq"),
    ("QALD-9_graph_exploration_evaluation.jsonl", "qald9"),
    ("QALD-10_graph_exploration_evaluation.jsonl", "qald10"),
    ("LC-QuAD_graph_exploration_evaluation.jsonl", "lcquad"),
    ("SPINACH_graph_exploration_evaluation.jsonl", "spinach")
]

for dataset_file, dataset_name in dataset_names_paths:
    print(f"Dataset: {dataset_name}")
    
    with open(dataset_file, "r") as f:
        dataset = json.load(f)
    
    path_type_counts = calculate_path_types(dataset)
    total_paths = sum(path_type_counts.values())
    
    for path_type, count in path_type_counts.items():
        percentage = (count / total_paths * 100) if total_paths > 0 else 0
        print(f"  {path_type}: {count} ({percentage:.2f}%)")
    print()