import json
import time
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.baselines.qa.llm_inherent import LlmQA
from src.engine.class_identifier.gold_class_identifier import GoldClassIdentifier
from src.engine.entity_linking.gold_entity_identifier import GoldEntityLinker
from src.engine.predicate_linking.gold_predicate_identifier import GoldPredicateIdentifier
from src.engine.config import CONFIG
from src.datasets.dataset import Dataset, DatasetFactory
from src.utils import execute_sparql_query, get_kgaqa_tracker, get_relative_path, endpoints_fill_parse_args
from src.logging import create_logger, log, LoggingOptions, LogLevel, create_console_logger
from src.engine.qa.query_generator.query_db import QueryDb
from src.baselines.sparql.llm_sparql_entities import LlmSparqlEntities
from tqdm import tqdm
import argparse
import os
from enum import Enum

    
# -----------------------------
# ----- Logging Functions -----
# -----------------------------
    
import tempfile
    
def atomic_write(filepath, data, mode='w'):
    dir_name = os.path.dirname(filepath)
    with tempfile.NamedTemporaryFile(mode=mode, dir=dir_name, delete=False) as tmp_file:
        json.dump(data, tmp_file, indent=4)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_name = tmp_file.name
    os.replace(temp_name, filepath)  # atomic rename

def save_to_file(run_results, output_file_path):    
    try:
        # Save run results
        atomic_write(output_file_path, run_results)
        # print(f"Saved run results to {output_file_path}")

    except Exception as e:
        print(f"Error saving files: {str(e)}")


if __name__ == "__main__":        
    create_console_logger()
    
    # ----------------------------------
    # ----- Command Line Arguments -----
    # ----------------------------------
    
    parser = argparse.ArgumentParser(
        description="Perform evaluation for generated file."
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output.")
    
    endpoints_fill_parse_args(parser)
    
    args = parser.parse_args()
    
    CONFIG(args)
    
    experiments = [
        # (output_name, dataset_name, dataset_path, index, query_db_name, query_db_path, entities_path)
        ("webqsp_oracle", "webqsp", "./datasets/webqsp/data/WebQSP.test_clean.json", "freebase", "webqsp", "./datasets/webqsp/data/WebQSP.train.json", "./datasets/webqsp/data/WebQSP-test-topic-entities.json"),
        ("webqsp_bela", "webqsp", "./datasets/webqsp/data/WebQSP.test_clean.json", "freebase", "webqsp", "./datasets/webqsp/data/WebQSP.train.json", "./logs-and-results/nerd/baselines/bela_webqsp_topic_entities.json"),
        ("cwq_oracle", "cwq", "./datasets/cwq/cwq_test.json", "freebase", "cwq", "./datasets/cwq/cwq_train.json", "./datasets/cwq/cwq_test_topic_entities.json"),
        ("cwq_bela", "cwq", "./datasets/cwq/cwq_test.json", "freebase", "cwq", "./datasets/cwq/cwq_train.json", "./logs-and-results/nerd/baselines/bela_cwq_topic_entities.json"),
        ("qald9_oracle", "qald-9", "./datasets/qald_9/qald_9_test.json", "dbpedia10", "qald-9", "./datasets/qald_9/qald-9-train-multilingual.json", "./datasets/qald_9/qald_9_test_topic_entities.json"),
        ("qald9_bela", "qald-9", "./datasets/qald_9/qald_9_test.json", "dbpedia10", "qald-9", "./datasets/qald_9/qald-9-train-multilingual.json", "./logs-and-results/nerd/baselines/bela_qald9_topic_entities.json"),
        ("lc_quad_oracle", "lc-quad-1", "./datasets/lc_quad_1/lc_quad_1_test-data.json", "dbpedia2016", "lc-quad-1", "./datasets/lc_quad_1/lc_quad_1_train-data.json", "./datasets/lc_quad_1/lc_quad_1_test-data_topic_entities_new.json"),
        ("lc_quad_bela", "lc-quad-1", "./datasets/lc_quad_1/lc_quad_1_test-data.json", "dbpedia2016", "lc-quad-1", "./datasets/lc_quad_1/lc_quad_1_train-data.json", "./logs-and-results/nerd/baselines/bela_lc_quad_topic_entities.json"),
        ("qald10_oracle", "qald-10", "./datasets/qald_10/qald_10.json", "wikidata", "qald-10", "./datasets/qald_10/qald_9_plus_train_wikidata.json", "./datasets/qald_10/qald_10_topic_entities.json"),
        ("qald10_bela", "qald-10", "./datasets/qald_10/qald_10.json", "wikidata", "qald-10", "./datasets/qald_10/qald_9_plus_train_wikidata.json", "./logs-and-results/nerd/baselines/bela_qald10_topic_entities.json"),
    ]
    
    for experiment in experiments:
        output_name, dataset_name, dataset_path, index, query_db_name, query_db_path, entities_path = experiment
        
        print(f"Running experiment: {output_name}...")
        
        # ------------------------
        # ----- Load Dataset -----
        # ------------------------

        dataset: Dataset = DatasetFactory.create_dataset(dataset_name, dataset_path)
        
        # -------------------------------
        # ----- Load Topic Entities -----
        # -------------------------------
        
        if not os.path.exists(entities_path):
            print(f"Entities file {entities_path} not found. Skipping experiment {output_name}.")
            continue
        
        with open(entities_path, 'r') as f:
            topic_entities_data = json.load(f)

        # -----------------------------------------------------
        # ----- Safeguard from overwriting existing files -----
        # -----------------------------------------------------
        
        output_file_path = os.path.join(get_relative_path(args.output_dir), output_name + "_generated.json")
        output_file_full_path = output_file_path.replace("_generated.json", "_generated_full.json")
        
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        if os.path.exists(get_relative_path(output_file_path)) == True:
            print(f"Generated file {output_file_path} already exists.")
            continue
        
        # -------------------------------
        # ----- Load Query Database -----
        # -------------------------------
        
        query_db_dataset = DatasetFactory.create_dataset(query_db_name, query_db_path)
        query_db = QueryDb(query_db_dataset)
        print("Loaded query database...")
        
        # --------------------------------
        # ----- Load Knowledge Graph -----
        # --------------------------------
        
        dataset.get_knowledge_graph().load(os.path.join(CONFIG().get("index_dir"), index))
        
        # ------------------------
        # ----- Setup System -----
        # ------------------------
        
        system = LlmSparqlEntities(
            model="gpt-4.1-mini",
            kg_name=dataset.get_knowledge_graph().name,
            kg=dataset.get_knowledge_graph(),
            use_cot=False,
            use_few_shot=True,
            query_db=query_db
        )
            
        results = []
        results_full = []
            
        for idx in tqdm(range(len(dataset))):    
            entry = dataset[idx]
            question = dataset.get_question(entry)
            topic_entities = topic_entities_data[idx]['topic_entities']
            
            record = None
            record_full = None
            
            # system_type is a SupportedSystems enum. Check against enum members.
            time_start = time.time()
            kwargs = {"query": dataset.get_query(entry), 
                      "topic_entities": topic_entities}
            generated_sparql, messages, metrics = system.generate_sparql(question, **kwargs)
            time_end = time.time()
            record = {
                "question": question,
                "sparql": generated_sparql,
                "elapsed": time_end - time_start,
                "metrics": metrics.to_dict()
            }
            record_full = {
                "question": question,
                "sparql": generated_sparql,
                "messages": messages,
                "elapsed": time_end - time_start,
                "metrics": metrics.to_dict()
            }
            
            results.append(record)
            results_full.append(record_full)
            
        save_to_file(results, output_file_path)
        save_to_file(results_full, output_file_full_path)