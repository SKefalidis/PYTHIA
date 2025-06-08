import json
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze agent entity linking results.")
    parser.add_argument("data_file", type=str, help="Path to the JSON file containing agent entity linking results.")
    parser.add_argument("generated_file", type=str, help="Path to the generated file.")
    parser.add_argument("--bela_output", type=str, default=None, help="Path to BELA output file (optional).")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"Data file {args.data_file} does not exist.")
        exit(1)
    
    dataset_data = {} # question: content
    with open(args.data_file, "r") as f:
        data = json.load(f)
        for entry in data:
            dataset_data[entry["question"]] = entry
        
    if not os .path.exists(args.generated_file):
        print(f"Generated file {args.generated_file} does not exist.")
        exit(1)
        
    with open(args.generated_file, "r") as f:
        generated = json.load(f)
        
    if args.bela_output is not None:
        if not os.path.exists(args.bela_output):
            print(f"BELa output file {args.bela_output} does not exist.")
            exit(1)
        bela_data = {}
        with open(args.bela_output, "r") as f:
            bela_entries = json.load(f)
            for entry in bela_entries[1:]:  # Skip header
                bela_data[entry["question"]] = entry
                
        # Integrate BELA data into generated data
        for entry in generated:
            question = entry["question"]
            if question in bela_data:
                entry["bela_predictions"] = bela_data[question]["predictions"]

    total_questions = 0
    total_named_entities = 0
    total_classes = 0

    used_tp = 0
    used_fp = 0
    used_fn = 0
    used_em = 0
    used_all_tp = 0
    
    used_bela_tp = 0
    used_bela_fp = 0
    used_bela_fn = 0
    used_bela_em = 0
    used_bela_all_tp = 0
    
    found_tp = 0
    found_fp = 0
    found_fn = 0
    for entry in generated:
        question = entry["question"]
        print("Question:", question)
        
        if question not in dataset_data:
            print("No corresponding data found in the dataset.")
            continue
        total_questions += 1
        
        gold_named_entities = dataset_data[question]["named_entities"]
        total_named_entities += len(gold_named_entities)
        
        gold_classes = dataset_data[question]["classes"]
        total_classes += len(gold_classes)
        
        gold_uris = gold_named_entities + gold_classes
        # print("Expected Entity Linking:")
        # print(f"- Named Entities: {gold_named_entities}")
        # print(f"- Classes: {gold_classes}")
        
        found_named_entities = entry["found_entities"]
        used_named_entities = entry["used_entities"]
        
        found_classes = entry["found_classes"]
        used_classes = entry["used_classes"]
        
        found_uris = found_named_entities + found_classes
        used_uris = used_named_entities + used_classes
        # print("Generated Entity Linking:")
        # print(f"- Named Entities: {found_named_entities}")
        # print(f"- Used Named Entities: {used_named_entities}")
        # print(f"- Classes: {found_classes}")
        # print(f"- Used Classes: {used_classes}")
        
        if set(used_uris) == set(gold_uris):
            used_em += 1
        if set(gold_uris).issubset(set(used_uris)):
            used_all_tp += 1
        
        for entity in gold_uris:
            if entity in used_uris:
                used_tp += 1
            else:
                print("Missing used entity:", entity)
                used_fn += 1
            if entity in found_uris:
                found_tp += 1
            else:
                found_fn += 1
        for entity in found_uris:
            if entity not in gold_uris:
                found_fp += 1
        for entity in used_uris:
            if entity not in gold_uris:
                used_fp += 1
        
    print("Total Questions:", total_questions)
    print("Total Named Entities:", total_named_entities)
    print("Total Classes:", total_classes)
    print("Used Entities - TP:", used_tp, "FP:", used_fp, "FN:", used_fn)
    print("Found Entities - TP:", found_tp, "FP:", found_fp, "FN:", found_fn)
    used_entities_precision = used_tp / (used_tp + used_fp) if (used_tp + used_fp) > 0 else 0.0
    used_entities_recall = used_tp / (used_tp + used_fn) if (used_tp + used_fn) > 0 else 0.0
    used_entities_f1 = 2 * (used_entities_precision * used_entities_recall) / (used_entities_precision + used_entities_recall) if (used_entities_precision + used_entities_recall) > 0 else 0.0
    print("Used Entities - Precision:", used_entities_precision, "Recall:", used_entities_recall, "F1:", used_entities_f1)
    print("Used Exact Match:", used_em, "out of", total_questions)
    print("Used All Correct Entities:", used_all_tp, "out of", total_questions)
    
    
    if "bela_predictions" in generated[0]:
        for entry in generated:
            question = entry["question"]
            if question not in dataset_data:
                # print("No corresponding data found in the dataset.")
                continue
            
            gold_classes = dataset_data[question]["classes"]
            gold_named_entities = dataset_data[question]["named_entities"]
            gold_uris = gold_classes + gold_named_entities
            
            entities = entry["bela_predictions"] + entry["used_classes"] + entry["used_entities"]
            
            if set(entities) == set(gold_uris):
                used_bela_em += 1
            if set(gold_uris).issubset(set(entities)):
                used_bela_all_tp += 1
            
            for entity in gold_uris:
                if entity in entities:
                    used_bela_tp += 1
                else:
                    used_bela_fn += 1
            for entity in entities:
                if entity not in gold_uris:
                    used_bela_fp += 1
                    
        print("BELA Named Entities - TP:", used_bela_tp, "FP:", used_bela_fp, "FN:", used_bela_fn)
        bela_entities_precision = used_bela_tp / (used_bela_tp + used_bela_fp) if (used_bela_tp + used_bela_fp) > 0 else 0.0
        bela_entities_recall = used_bela_tp / (used_bela_tp + used_bela_fn) if (used_bela_tp + used_bela_fn) > 0 else 0.0
        bela_entities_f1 = 2 * (bela_entities_precision * bela_entities_recall) / (bela_entities_precision + bela_entities_recall) if (bela_entities_precision + bela_entities_recall) > 0 else 0.0
        print("BELA Named Entities - Precision:", bela_entities_precision, "Recall:", bela_entities_recall, "F1:", bela_entities_f1)
        print("BELA Exact Match:", used_bela_em, "out of", total_questions)
        print("BELA All Correct Entities:", used_bela_all_tp, "out of", total_questions)
    