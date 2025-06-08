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
from src.logging import create_logger, log, LoggingOptions, LogLevel, LogComponent, LogType, logging_fill_parse_args, logging_set_from_args
from src.engine.qa.query_generator.query_db import QueryDb
from src.baselines.sparql.llm_sparql import LlmSparqlBaseline
from src.baselines.sparql.llm_sparql_gold import LlmSparqlGoldBaseline
from tqdm import tqdm
import argparse
import os
from enum import Enum


class SupportedSystems(Enum):
    SPARQL = 1
    SPARQL_GOLD = 2
    LLM_INHERENT = 3
    LLM_WEB = 4
    
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
        
def generate_and_evaluate_sparql_queries(dataset_type: str, dataset_path: str, index: str,
                                         system_type: SupportedSystems, model: str, use_cot: bool, use_few_shot: bool, query_db_path: str, 
                                         output_directory: str,
                                         debug: bool = False) -> int:
    # ------------------------
    # ----- Load Dataset -----
    # ------------------------

    dataset: Dataset = DatasetFactory.create_dataset(dataset_type, dataset_path)
    

    # -----------------------------------------------------
    # ----- Safeguard from overwriting existing files -----
    # -----------------------------------------------------
    
    output_file_path = os.path.join(get_relative_path(output_directory), system_type.name.lower(), model, f"{dataset_type}{'_cot' if use_cot else ''}{'_fewshot' if use_few_shot else ''}_generated.json")
    output_file_full_path = output_file_path.replace("_generated.json", "_generated_full.json")
    
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    if os.path.exists(get_relative_path(output_file_path)) == True:
        print(f"Generated file {output_file_path} already exists.")
        return 1
    
    # -------------------------------
    # ----- Load Query Database -----
    # -------------------------------
    
    if query_db_path and use_few_shot:
        query_db_dataset = DatasetFactory.create_dataset(dataset_type, query_db_path)
        query_db = QueryDb(query_db_dataset)
        print("Loaded query database...")
    else:
        if use_few_shot:
            print("Few-shot prompting requested but no query database provided. Skipping...")
            return 1
        query_db = None
        print("No query database provided...")
    
    # ------------------------
    # ----- Setup System -----
    # ------------------------
    
    system: LlmSparqlBaseline|LlmQA = None
    if  system_type == SupportedSystems.SPARQL:
        system = LlmSparqlBaseline(model, dataset.get_knowledge_graph()._name_, dataset.get_knowledge_graph(), use_cot, use_few_shot, query_db)
    elif system_type == SupportedSystems.SPARQL_GOLD:
        gold_entity_identifier = GoldEntityLinker(dataset.get_knowledge_graph(), dataset.get_prefixes())
        gold_class_identifier = GoldClassIdentifier(dataset.get_knowledge_graph(), dataset.get_knowledge_graph().endpoint, dataset.get_prefixes())
        gold_predicate_identifier = GoldPredicateIdentifier(dataset.get_prefixes(), dataset.get_knowledge_graph())
        system = LlmSparqlGoldBaseline(model, dataset.get_knowledge_graph()._name_, use_cot, use_few_shot, query_db, dataset.get_knowledge_graph(), gold_entity_identifier, gold_class_identifier, gold_predicate_identifier)
    elif system_type == SupportedSystems.LLM_INHERENT:
        year = 2024
        if dataset.get_knowledge_graph() == KnowledgeGraphs.WIKIDATA:
            year = 2022
        elif dataset.get_knowledge_graph() == KnowledgeGraphs.DBPEDIA or dataset.get_knowledge_graph() == KnowledgeGraphs.DBPEDIA10:
            year = 2016
        elif dataset.get_knowledge_graph() == KnowledgeGraphs.FREEBASE:
            year = 2015
        system = LlmQA(model, use_cot, year=year)
    elif system_type == SupportedSystems.LLM_WEB:
        raise ValueError(f"Unsupported system: {system_type}")
    else:
        raise ValueError(f"Unknown system: {system_type}")
        
    results = []
    results_full = []
        
    for idx in tqdm(range(len(dataset))):    
        if debug: 
            print(f"Answering question {idx+1} of {len(dataset)}")
        entry = dataset[idx]
        question = dataset.get_question(entry)
        
        record = None
        record_full = None
        
        # system_type is a SupportedSystems enum. Check against enum members.
        if system_type in (SupportedSystems.SPARQL, SupportedSystems.SPARQL_GOLD):
            time_start = time.time()
            kwargs = {"query": dataset.get_query(entry)} if system_type == SupportedSystems.SPARQL_GOLD else {}
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
        elif system_type == SupportedSystems.LLM_INHERENT:
            time_start = time.time()
            generated_answer, messages, metrics = system.generate(question)
            time_end = time.time()
            record = {
                "question": question,
                "answer": generated_answer,
                "elapsed": time_end - time_start,
                "metrics": metrics.to_dict()
            }
            record_full = {
                "question": question,
                "answer": generated_answer,
                "messages": messages,
                "elapsed": time_end - time_start,
                "metrics": metrics.to_dict()
            }
        else:
            raise ValueError(f"Unsupported system: {system_type}")
        
        results.append(record)
        results_full.append(record_full)
        
    save_to_file(results, output_file_path)
    save_to_file(results_full, output_file_full_path)
    
    # from src.evaluation.qa_engine.sparql_eval import evaluate_generated_file
    # evaluate_generated_file(dataset, output_file_path)
    
    return 0


if __name__ == "__main__":    
    
    import sys
    print("ARGV:", sys.argv)
    
    create_logger("generate_results_hidden", ".", LoggingOptions.LOG_TO_CONSOLE, LogLevel.INFO)
    
    # ----------------------------------
    # ----- Command Line Arguments -----
    # ----------------------------------
    
    parser = argparse.ArgumentParser(
        description="Perform evaluation for generated file."
    )
    
    subparsers = parser.add_subparsers(
        required=False,
        dest="subprogram",
        help="Subprograms"
    )
    
    # -----------------------------------
    # ----- Pre-written Experiments -----
    # -----------------------------------
    
    parser_prewritten = subparsers.add_parser('prewritten', help='Pre-written experiment commands')
    
    parser_prewritten.add_argument("--output_dir", type=str, required=True, help="Path to save the output.")
    parser_prewritten.add_argument("--sparql", action='store_true', help="Whether to run SPARQL experiments.")
    parser_prewritten.add_argument("--sparql_gold", action='store_true', help="Whether to run SPARQL-GOLD experiments.")
    parser_prewritten.add_argument("--llm_inherent", action='store_true', help="Whether to run LLM-INHERENT experiments.")
    parser_prewritten.add_argument("--bedrock", action='store_true', help="Whether to use Bedrock models.")
    parser_prewritten.add_argument("--llama", action='store_true', help="Whether to use LLaMA models.")
    parser_prewritten.add_argument("--oss", action='store_true', help="Whether to use Open Source models.")
    parser_prewritten.add_argument("--openai", action='store_true', help="Whether to use OpenAI models.")
    
    endpoints_fill_parse_args(parser_prewritten)
    
    # -----------------------------
    # ----- Custom Experiment -----
    # -----------------------------
    
    parser_custom = subparsers.add_parser('custom', help='Custom-related commands')
    
    parser_custom.add_argument("--system", type=str, required=True, choices=[system.name for system in SupportedSystems], help="Which models to use.")
    
    parser_custom.add_argument("--llm_model", type=str, required=True, help="LLM model to use for LLM-based systems.")
    
    parser_custom.add_argument("--cot", action='store_true', help="Whether to use chain-of-thought prompting.")
    parser_custom.add_argument("--few_shot", action='store_true', help="Whether to use few-shot prompting.")
    
    parser_custom.add_argument("--query_db_file", type=str, required=False, help="Path to the query database file (optional)")
    
    parser_custom.add_argument("--output_dir", type=str, required=True, help="Path to save the output.")
    
    DatasetFactory.fill_parse_args(parser_custom)
    
    endpoints_fill_parse_args(parser_custom)
    
    # -------------------
    # ----- Parsing -----
    # -------------------
    
    args = parser.parse_args()
        
    if args.subprogram == "custom":
        # Initialize configuration with parser arguments
        CONFIG(args)        
        generate_and_evaluate_sparql_queries(args.dataset, args.dataset_path,
                                            SupportedSystems[args.system], args.llm_model, args.cot, args.few_shot, args.query_db_file,
                                            args.output_dir)
    elif args.subprogram == "prewritten":
        # Initialize configuration with parser arguments
        CONFIG(args)
        
        # Experiments
        datasets = [
            ("qald-9", get_relative_path("../../../datasets/qald_9/qald_9_test.json"), get_relative_path("../../../datasets/qald_9/qald-9-train-multilingual.json"), "dbpedia10"),
            ("qald-10", get_relative_path("../../../datasets/qald_10/qald_10.json"), get_relative_path("../../../datasets/qald_9_plus/qald_9_plus_train_wikidata.json"), "wikidata"),
            ("lc-quad-1", get_relative_path("../../../datasets/lc_quad_1/lc_quad_1_test-data.json"), get_relative_path("../../../datasets/lc_quad_1/lc_quad_1_train-data.json"), "dbpedia2016"),
            ("lc-quad-2", get_relative_path("../../../datasets/lc_quad_2/lc_quad_2_test.json"), get_relative_path("../../../datasets/lc_quad_2/lc_quad_2_train.json"), "wikidata"),
            ("spinach", get_relative_path("../../../datasets/spinach/test_clean.json"), None, "wikidata"),
            ("webqsp", get_relative_path("../../../datasets/webqsp/data/WebQSP.test_clean.json"), get_relative_path("../../../datasets/webqsp/data/WebQSP.train.json"), "freebase"),
            ("cwq", get_relative_path("../../../datasets/cwq/cwq_test.json"), get_relative_path("../../../datasets/cwq/cwq_train.json"), "freebase"),
            ("geoq1089", get_relative_path("../../../datasets/geoq1089/GeoQuestions1089.json"), None, "yago2geo"),
            ("bestiary", get_relative_path("../../../datasets/beastiary/beastiary_with_qald_format.json"), None, "beastiary"),
        ]
        
        system_configurations = []

        system_configurations_sparql = [
            (SupportedSystems.SPARQL, False, False),        # No CoT, No Few-Shot
            # (SupportedSystems.SPARQL, True, False),         # CoT, No Few-Shot
            (SupportedSystems.SPARQL, False, True),         # No CoT, Few-Shot
            # (SupportedSystems.SPARQL, True, True),          # CoT, Few-Shot
        ]

        system_configurations_gold = [
            (SupportedSystems.SPARQL_GOLD, False, False),   # No CoT, No Few-Shot
            # (SupportedSystems.SPARQL_GOLD, True, False),    # CoT, No Few-Shot
            (SupportedSystems.SPARQL_GOLD, False, True),    # No CoT, Few-Shot
            # (SupportedSystems.SPARQL_GOLD, True, True),     # CoT, Few-Shot
        ]

        system_configurations_inherent = [
            (SupportedSystems.LLM_INHERENT, False, False),
            (SupportedSystems.LLM_INHERENT, True, True)
        ]

        if args.sparql:
            system_configurations.extend(system_configurations_sparql)
        if args.sparql_gold:
            system_configurations.extend(system_configurations_gold)
        if args.llm_inherent:
            system_configurations.extend(system_configurations_inherent)
        if not system_configurations:
            raise ValueError("No system configurations selected for pre-written experiments.")
        
        models = []

        if args.bedrock:
            models.extend([
                "bedrock/openai.gpt-oss-20b-1:0",
                "bedrock/openai.gpt-oss-120b-1:0",
                "bedrock/meta.llama3-8b-instruct-v1:0",
                "bedrock/meta.llama3-70b-instruct-v1:0",
            ])
        if args.llama:
            models.extend([
                "bedrock/meta.llama3-8b-instruct-v1:0",
                "bedrock/meta.llama3-70b-instruct-v1:0",
            ])
        if args.oss:
            models.extend([
                "bedrock/openai.gpt-oss-20b-1:0",
                "bedrock/openai.gpt-oss-120b-1:0",
            ])
        if args.openai:
            models.extend([
                "gpt-4.1-mini",
                # "gpt-4.1",
                # "gpt-5.1"
            ])
        if not models:
            raise ValueError("No models selected for pre-written experiments.")
        
        for dataset_name, dataset_path, query_db_path, index in datasets:
            for system_type, use_cot, use_few_shot in system_configurations:
                for model in models:
                    print(f"Running experiment: Dataset={dataset_name}, System={system_type.name}, Model={model}, CoT={use_cot}, Few-Shot={use_few_shot}")
                    time_start_exp = time.time()
                    generate_and_evaluate_sparql_queries(dataset_name, dataset_path, index,
                                                        system_type, model, use_cot, use_few_shot, query_db_path,
                                                        args.output_dir)
                    print(f"Experiment completed in {time.time() - time_start_exp:.2f} seconds.\n")
