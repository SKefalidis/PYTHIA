# import argparse
# import re
# import os
# import json
# from SPARQLWrapper import SPARQLWrapper, JSON
# from typing import List

# from src.datasets.dataset import KnowledgeGraph, uris_to_urils, urils_to_uris, triples_with_urils_to_triples_with_uris, triples_with_uris_to_triples_with_urils
# from src.engine.qa.query_db import QueryDb
# from src.utils import SupportedLLMs, get_relative_path, llm_call
# from src.logging import LoggingOptions, create_logger, log, LogComponent, LogLevel, LogType, print_colored

# from src.engine.entity_linking.gold_entity_identifier import GoldEntityLinker
# from src.engine.class_identifier.gold_class_identifier import GoldClassIdentifier

# from src.datasets.lc_quad_1_dataset import LcQuad1Dataset
# from src.datasets.lc_quad_2_dataset import LcQuad2Dataset
# from src.datasets.cwq_dataset import CwqDataset
# from src.datasets.webqsp_dataset import WebQSPDataset
# from src.datasets.qald10_dataset import Qald10Dataset
# from src.datasets.qald9_dataset import Qald9Dataset
# from src.engine.gost_requests import validate_query

# from src.engine.qa.query_generator import compare_queries_loose, compute_metrics, query_has_results


# PROMPT_FEWSHOT_QUERY_GENERATION_ENTRY = """
# You are an expert in generating SPARQL queries based on natural language questions.

# You will be given:
# - a question,
# - a list of entities from a Knowledge Graph,
# - a list of classes from the ontology of the same Knowledge Graph,
# - a few examples of similar questions and their corresponding SPARQL queries.

# Your task is twofold:

# 1. Determine if it is possible to generate a SPARQL query that answers the question using only:
#    - URIs present in the provided entities and classes, or
#    - URIs used in the example SPARQL queries.
   
#    You must not invent or use any URIs not explicitly given.

#    A query can only be generated if there is a **very close match** between the incoming question and one of the examples — i.e., they are nearly identical in both structure and intent.
   
#    If that condition is met, adapt the example accordingly to generate a new query using only allowed URIs. Include only what is necessary — not all entities and classes must be used.
   
#    Do not use prefixes in your query. Please use full URIs for all entities, classes, and predicates/relations.
   
#    Output the SPARQL query inside triple backticks (```).

# 2. If no sufficiently similar example exists, or the required entities/classes are missing, respond with:
#    `UNABLE_TO_GENERATE_QUERY`

# Be precise and cautious.

# ---

# Question: {question}
# Entities: {entities}
# Classes: {classes}
# Examples:
# {examples}
# """

# PROMPT_FEWSHOT_QUERY_GENERATION_FULL = """
# You are an expert in generating SPARQL queries based on natural language questions.

# You will be given:
# - a question,
# - a list of entities from a Knowledge Graph,
# - a list of classes from the ontology of the same Knowledge Graph,
# - a few examples of similar questions and their corresponding SPARQL queries.

# Your task is to generate a SPARQL query that answers the question/

# Do not use prefixes in your query. Please use full URIs for all entities, classes, and predicates/relations.   

# Output the SPARQL query inside triple backticks (```).

# Be precise and cautious.

# ---

# Question: {question}
# Entities: {entities}
# Classes: {classes}
# Examples:
# {examples}
# """



# class EntryQueryGenerator:
#     """
#     Generates entry queries for the QA system.
#     """

#     def __init__(self, model_id: SupportedLLMs, query_db = None):
#         self.model_id = model_id
#         self.query_db: QueryDb = query_db
        

#     def extract_sparql(self, text):
#         pattern = r"```sparql(.*?)```"
#         matches = re.findall(pattern, text, re.DOTALL)
#         if matches:
#             return matches[-1].strip()
        
#         pattern = r"```(.*?)```"
#         matches = re.findall(pattern, text, re.DOTALL)
#         if matches:
#             return matches[-1].strip()
        
#         return text

#     def generate(self, question: str, entities, classes, prompt: str):      
#         relevant_questions, relevant_queries = self.query_db.get_relevant_queries(question, top_k=5) # TODO: Make get relevant queries return URILs instead of URIs.
#         prompt = prompt.format(
#             question=question,
#             entities=", ".join(entities),
#             classes=", ".join(classes),
#             examples="\n\n".join([
#                 f"Question: {q}\nQuery: {r}" for q, r in zip(relevant_questions, relevant_queries)
#             ]),
#             )
#         log(f"[ENTRY] Prompt: {prompt}", LogComponent.QUERY_GENERATOR, LogLevel.DEBUG, LogType.PROMPT)
        
#         generated = llm_call(self.model_id, prompt, max_tokens=512)
#         log(f"[ENTRY] Generated: {generated}", LogComponent.QUERY_GENERATOR, LogLevel.DEBUG, LogType.LLM_RESULT)
        
#         if "UNABLE_TO_GENERATE_QUERY" in generated:
#             log(f"[ENTRY] Unable to generate query for question: {question}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
#             return "UNABLE_TO_GENERATE_QUERY"
                
#         query = self.extract_sparql(generated)
#         log(f"[ENTRY] Generated query: {query}", LogComponent.QUERY_GENERATOR, LogLevel.DEBUG, LogType.LLM_RESULT)
        
#         return query
    

# if __name__ == "__main__":    
    
#     # ----------------------------------
#     # ----- Command Line Arguments -----
#     # ----------------------------------
    
#     parser = argparse.ArgumentParser(description="Process QA dataset and run queries.")
    
#     parser.add_argument("--test_file", type=str, required=True,
#                         help="Path to the test file (e.g., QALD or WebQSP)")
    
#     parser.add_argument("--query_db_file", type=str, required=False,
#                     help="Path to the query database file (optional)")
    
#     parser.add_argument("--results_dir", type=str, required=True, default='./results/',
#                         help="Path to the directory where the results will be stored")
    
#     parser.add_argument("--metrics_dir", type=str, required=True, default='./results/',
#                         help="Path to the directory where the metrics will be stored")
        
#     parser.add_argument("--logs_dir", type=str, required=True,
#                         help="Directory where logs will be saved")
    
#     parser.add_argument("--mode", type=str, required=True, choices=["full", "entry"],
#                         help="Whether to use the full query generator or the entry query generator. Use 'full' for full queries and 'entry' for entry queries.")
    
#     parser.add_argument("--endpoint", type=int, required=True, default='1',
#                         help="Which endpoint to use, choose from (1, 2, 3)")

#     args = parser.parse_args()
    
#     # ------------------------
#     # ----- Load Dataset -----
#     # ------------------------

#     if "webqsp" in args.test_file.lower():
#         dataset = WebQSPDataset.from_files(args.test_file)
#     elif "cwq" in args.test_file.lower():
#         dataset = CwqDataset.from_files(args.test_file)
#     elif "qald_9" in args.test_file.lower():
#         dataset = Qald9Dataset.from_files(args.test_file)
#     elif "lc_quad_1" in args.test_file.lower():
#         dataset = LcQuad1Dataset.from_files(args.test_file)
#     elif "qald_10" in args.test_file.lower():
#         dataset = Qald10Dataset.from_files(args.test_file)
#     elif "lc_quad_2" in args.test_file.lower():
#         dataset = LcQuad2Dataset.from_files(args.test_file)
#     else:
#         raise ValueError("Unrecognized dataset format in test_file path.")
    
#     # -------------------------------
#     # ----- Load Query Database -----
#     # -------------------------------
    
#     if "webqsp" in args.query_db_file.lower():
#         query_db_dataset = WebQSPDataset.from_files(args.query_db_file)
#     elif "cwq" in args.query_db_file.lower():
#         query_db_dataset = CwqDataset.from_files(args.query_db_file)
#     elif "qald_9" in args.query_db_file.lower():
#         query_db_dataset = Qald9Dataset.from_files(args.query_db_file)
#     elif "lc_quad_1" in args.query_db_file.lower():
#         query_db_dataset = LcQuad1Dataset.from_files(args.query_db_file)
#     elif "qald_10" in args.query_db_file.lower():
#         query_db_dataset = Qald10Dataset.from_files(args.query_db_file)
#     elif "lc_quad_2" in args.query_db_file.lower():
#         query_db_dataset = LcQuad2Dataset.from_files(args.query_db_file)
#     else:
#         raise ValueError("Unrecognized dataset format in query_db_file path.")
    
#     # -------------------------
#     # ----- Setup Logging -----
#     # -------------------------

#     os.makedirs(get_relative_path(args.logs_dir), exist_ok=True)
#     create_logger("entry_qg_" + args.mode + "_" + args.test_file.lower().split('/')[-1], args.logs_dir, log_option=LoggingOptions.LOG_TO_BOTH, log_level=LogLevel.DEBUG)
    
#     # ----------------------------------
#     # ----- Setup Metrics Tracking -----
#     # ----------------------------------
    
#     os.makedirs(get_relative_path(args.results_dir), exist_ok=True)
#     os.makedirs(get_relative_path(args.metrics_dir), exist_ok=True)
    
#     performance_metrics = {}
    
#     # ------------------------------
#     # ----- Main Functionality -----
#     # ------------------------------
    
#     generation_prompt = PROMPT_FEWSHOT_QUERY_GENERATION_ENTRY if args.mode == "entry" else PROMPT_FEWSHOT_QUERY_GENERATION_FULL
    
#     import src.datasets.dataset
#     src.datasets.dataset.TRIPLESTORE_ENDPOINT = args.endpoint
#     log(f"Using endpoint {src.datasets.dataset.TRIPLESTORE_ENDPOINT}", LogComponent.KNOWLEDGE_BASE, LogLevel.INFO, LogType.NORMAL)
    
#     kg = dataset.get_knowledge_graph()
    
#     entity_linker = GoldEntityLinker(knowledge_graph=kg, prefixes=dataset.get_prefixes())
#     class_identifier = GoldClassIdentifier(knowledge_graph=kg, endpoint_url=KnowledgeGraph.get_endpoint(kg), prefixes=dataset.get_prefixes())
    
#     query_db = QueryDb(query_db_dataset)
#     generator = EntryQueryGenerator(model_id=SupportedLLMs.GPT4_1_MINI, query_db=query_db)
    
#     run_results = []
    
#     if "qald" in args.test_file:
#         qald_json = json.load(open(args.test_file, 'r', encoding='utf-8'))
#     else:
#         qald_json = None
        
#     total = 0
#     no_generation = 0 # the model decided that it cannot generate a query
#     empty_generation = 0 # a query was generated, but it returns no results
#     invalid_gold_queries = 0
#     empty_gold_queries = 0
#     exact_match = 0 # the generated query is exactly the same as the gold query
#     total_tp = 0 # total true positives
#     total_fp = 0 # total false positives
#     total_fn = 0 # total false negatives
#     total_hits_at_1 = 0 # total hits at 1
#     total_macro_f1 = 0 # total macro F1
#     total_macro_precision = 0 # total macro precision
#     total_macro_recall = 0 # total macro recall

#     for idx, entry in enumerate(dataset):      
#         total += 1
          
#         question = dataset.get_question(entry)
#         gold_query = dataset.get_query(entry)
        
#         # Fix missing prefixes
#         fixed_gold_query = ""
#         prefixes = dataset.get_prefixes().split("\n")
#         for prefix in prefixes:
#             prefix = prefix.replace("\n", "").strip()
#             if prefix == "":
#                 continue
#             prefix_keyword, prefix_name, prefix_value = prefix.split(" ")
#             pattern = r'PREFIX\s+'+re.escape(prefix_name)
#             if re.search(pattern, gold_query) is None:
#                 fixed_gold_query += prefix + "\n"
#         fixed_gold_query += gold_query
#         gold_query = fixed_gold_query
        
#         # print progress
#         log(f"Progress: {idx+1}/{len(dataset)}", LogComponent.OTHER, LogLevel.INFO, LogType.HEADER)
#         log(f"Question: {question}", LogComponent.OTHER, LogLevel.INFO, LogType.HEADER)
        
#         # If the query can't be answered or is invalid, we generate to have valid results for Gerbil, but we don't count it in the metrics.
#         skip_metrics = False
#         if validate_query(gold_query) == False:
#             log(f"Invalid gold query: {gold_query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
#             invalid_gold_queries += 1
#             skip_metrics = True
#         if query_has_results(KnowledgeGraph.get_endpoint(kg), gold_query) == False:
#             log(f"Empty gold query: {gold_query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
#             empty_gold_queries += 1
#             skip_metrics = True
                
#         # ------------------------------------
#         # ----- Get entities and classes -----
#         # ------------------------------------
#         classes = class_identifier.identify(gold_query)
#         entities = entity_linker.identify(gold_query)
#         entities = [uri for uri in entities if uri not in classes] # FIXME: Temporary fix because the Wikidata gold entity linker returns classes as entities.
        
#         classes = uris_to_urils(classes, kg)
#         entities = uris_to_urils(entities, kg)
        
#         # print progress
#         log(f"Entities: {entities}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
#         log(f"Classes: {classes}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
        
#         # --------------------
#         # ----- Generate -----
#         # --------------------
#         query = generator.generate(question, entities, classes, generation_prompt)
#         if query == "UNABLE_TO_GENERATE_QUERY":
#             log(f"Unable to generate query for question: {question}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
#             no_generation += 1
#         else:
#             try:
#                 # print("Generated query:", query)
#                 fixed_query = ""
#                 prefixes = dataset.get_prefixes().split("\n")
#                 for prefix in prefixes:
#                     prefix = prefix.replace("\n", "").strip()
#                     if prefix == "":
#                         continue
#                     prefix_keyword, prefix_name, prefix_value = prefix.split(" ")
#                     pattern = r'PREFIX\s+'+re.escape(prefix_name)
#                     if re.search(pattern, query) is None:
#                         fixed_query += prefix + "\n"
#                 fixed_query += query
#                 query = fixed_query
#                 # print("fixed query:", query)
#                 query = triples_with_urils_to_triples_with_uris(query, kg)
#                 # print("query after triples_with_urils_to_triples_with_uris:", query)
                
#                 if validate_query(query) == True:
#                     if query_has_results(KnowledgeGraph.get_endpoint(kg), query, 0) == False:
#                         if args.mode == "entry":
#                             query = "UNABLE_TO_GENERATE_QUERY"
#                             empty_generation += 1
#                             log(f"Empty query generated: {query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
#                 else:
#                     if args.mode == "entry":
#                         log(f"Invalid generated query: {query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
#                         query = "UNABLE_TO_GENERATE_QUERY"
#                         no_generation += 1
#             except:
#                 log(f"Error in triples_with_urils_to_triples_with_uris: {query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
        
#         # ------------------------
#         # ----- Update Files -----
#         # ------------------------
        
#         if query != "UNABLE_TO_GENERATE_QUERY":
#             sparql = SPARQLWrapper(KnowledgeGraph.get_endpoint(kg))
#             sparql.setCredentials("user", "PASSWORD")
#             sparql.setQuery(query)
#             sparql.setReturnFormat(JSON)
#             try:
#                 results = sparql.query().convert()
#                 if qald_json is not None:
#                     qald_json['questions'][idx]['query']['sparql'] = query
#                     qald_json['questions'][idx]['answers'] = [results]
#             except Exception as e:
#                 results = ""
#                 if qald_json:
#                     qald_json['questions'][idx]['query']['sparql'] = query
#                     qald_json['questions'][idx]['answers'] = []
#                 log(f"Error in SPARQL query: {query}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.NORMAL)
#                 print(e)
#         else:
#             results = ""
#             if qald_json:
#                 qald_json['questions'][idx]['query']['sparql'] = query
#                 qald_json['questions'][idx]['answers'] = []
                
#         entry = {
#             "question": question,
#             "gold_query": gold_query,
#             "entities": entities,
#             "classes": classes,
#             "generated_query": query,
#             "answer": [results]
#         }
#         run_results.append(entry)
        
#         # --------------------------
#         # ----- Update Metrics -----
#         # --------------------------

#         if skip_metrics == False:
#             log(f"Gold query: {gold_query}", LogComponent.OTHER, LogLevel.INFO, LogType.GOLD)
#             log(f"Generated query: {query}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
            
#             if query != "UNABLE_TO_GENERATE_QUERY":
#                 tp, fp, fn, hits_at_1 = compare_queries_loose(question, KnowledgeGraph.get_endpoint(kg), query, gold_query)

#                 if tp > 0 and fp == 0 and fn == 0:
#                     log(f"✅ Correct 0-shot: {question}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
#                     exact_match += 1
#                 else:
#                     log(f"❌ Incorrect 0-shot: {question}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
                    
#                 log(f"{tp} TP, {fp} FP, {fn} FN", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
#                 log(f"Hits@1: {hits_at_1}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
#             else:
#                 tp, fp, fn, hits_at_1 = 0, 0, 0, 0
#                 log(f"⏩ Unable to generate query for: {question}", LogComponent.OTHER, LogLevel.INFO, LogType.NORMAL)
                    
#             total_tp += tp
#             total_fp += fp
#             total_fn += fn
#             total_hits_at_1 += hits_at_1
            
#             # Compute metrics
#             metrics = compute_metrics(tp, fp, fn)            
#             total_macro_f1 += metrics['f1']
#             total_macro_precision += metrics['precision']
#             total_macro_recall += metrics['recall']
            
#             log(f"\nMacro-averaged metrics: {metrics}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.HEADER)
#             log(f"\tTotal: {total}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
#             log(f"\tTotal skipped: {(no_generation+empty_generation)}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
            
#             log(f"\nAdjusted metrics", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.HEADER)
#             log(f"\tPrecision: {total_macro_precision/(total-no_generation-empty_generation):.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
#             log(f"\tRecall: {total_macro_recall/(total-no_generation-empty_generation):.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
#             log(f"\tF1: {total_macro_f1/(total-no_generation-empty_generation):.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
#             log(f"\tHits@1: {total_hits_at_1/(total-no_generation-empty_generation):.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
#             log(f"\tExact match: {exact_match/(total-no_generation-empty_generation):.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
            
#             log(f"\nFull metrics:", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.HEADER)
#             log(f"\tPrecision: {total_macro_precision/total:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
#             log(f"\tRecall: {total_macro_recall/total:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
#             log(f"\tF1: {total_macro_f1/total:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
#             log(f"\tHits@1: {total_hits_at_1/total:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
#             log(f"\tExact match: {exact_match/total:.2f}", LogComponent.OTHER, LogLevel.PERFORMANCE_UPDATES, LogType.NORMAL)
  
            
#     dataset_name = args.test_file.lower().split('/')[-1]

#     # Save results file
#     with open(os.path.join(get_relative_path(args.results_dir), f"run_results_entry_{args.mode}_{dataset_name}.json"), 'w') as f:
#         json.dump(run_results, f, indent=4)
#     log(f"Saved run results to {os.path.join(get_relative_path(args.results_dir), f'run_results_entry_{args.mode}_{dataset_name}.json')}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.HEADER)
    
#     # Save the performance metrics
#     metrics = {
#         "total": total,
#         "no_generation": no_generation,
#         "empty_generation": empty_generation,
#         "invalid_gold_queries": invalid_gold_queries,
#         "empty_gold_queries": empty_gold_queries,
#         "exact_match": exact_match,
#         "total_tp": total_tp,
#         "total_fp": total_fp,
#         "total_fn": total_fn,
#         "total_hits_at_1": total_hits_at_1,
#         "macro_f1": total_macro_f1 / total if total > 0 else 0,
#         "macro_precision": total_macro_precision / total if total > 0 else 0,
#         "macro_recall": total_macro_recall / total if total > 0 else 0
#     }
#     with open(os.path.join(get_relative_path(args.metrics_dir), f"metrics_entry_{args.mode}_{dataset_name}.json"), 'w') as f:
#         json.dump(metrics, f, indent=4)
#     log(f"Saved metrics to {os.path.join(get_relative_path(args.metrics_dir), f'metrics_entry_{args.mode}_{dataset_name}.json')}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.HEADER)
    
#     # Save the QALD JSON file
#     if qald_json is not None:
#         with open(os.path.join(get_relative_path(args.results_dir), f"qald_entry_{args.mode}_{dataset_name}.json"), 'w') as f:
#             json.dump(qald_json, f, indent=4)
#         log(f"Saved QALD JSON basic to {os.path.join(get_relative_path(args.results_dir), f'qald_entry_{args.mode}_{dataset_name}.json')}", LogComponent.QUERY_GENERATOR, LogLevel.INFO, LogType.HEADER)