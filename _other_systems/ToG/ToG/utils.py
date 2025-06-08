from prompt_list import *
import json
import time
import openai
import re
import os
from prompt_list import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

def retrieve_top_docs(query, docs, model, width=3):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query (str): The input query.
    - docs (list of str): The list of documents to search from.
    - model_name (str): The name of the SentenceTransformer model to use.
    - width (int): The number of top documents to return.

    Returns:
    - list of float: A list of scores for the topn documents.
    - list of str: A list of the topn documents.
    """

    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores


def compute_bm25_similarity(query, corpus, width=3):
    """
    Computes the BM25 similarity between a question and a list of relations,
    and returns the topn relations with the highest similarity along with their scores.

    Args:
    - question (str): Input question.
    - relations_list (list): List of relations.
    - width (int): Number of top relations to return.

    Returns:
    - list, list: topn relations with the highest similarity and their respective scores.
    """

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)
    
    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    doc_scores = sorted(doc_scores, reverse=True)[:width]

    return relations, doc_scores


def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations

from globals import *

def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    get_current_metrics().LLM_CALLS += 1
    start_time = time.time()

    if "llama" in engine.lower():
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"  # your local llama server port
        engine = openai.Model.list()["data"][0]["id"]
    else:
        openai_obj = openai.OpenAI(api_key=opeani_api_keys)

    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    f = 0
    while(f == 0):
        try:
            response = openai_obj.chat.completions.create(
                seed=451, # 0451
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0
            )
            result = response.choices[0].message.content
            usage = response.usage
            get_current_metrics().LLM_INPUTS += usage.prompt_tokens
            get_current_metrics().LLM_OUTPUTS += usage.completion_tokens
            f = 1
        except Exception as e:
            print("openai error, retry")
            print(e)
            time.sleep(2)
    get_current_metrics().LLM_TIME += time.time() - start_time
    return result

    
def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates


def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)
    

def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name):
    time_taken = time.time() - get_timer()
    get_current_metrics().TIME = time_taken

    dict = {
        "question":question, 
        "results": answer, 
        "reasoning_chains": cluster_chain_of_entities, 
        "metrics" : {
            "TIME": get_current_metrics().TIME,
            "SPARQL_CALLS": get_current_metrics().SPARQL_CALLS,
            "SPARQL_TIME": get_current_metrics().SPARQL_TIME,
            "LLM_CALLS": get_current_metrics().LLM_CALLS,
            "LLM_INPUTS": get_current_metrics().LLM_INPUTS,
            "LLM_OUTPUTS": get_current_metrics().LLM_OUTPUTS,
            "LLM_TIME": get_current_metrics().LLM_TIME
        }
    }
    
    entries = []
    if os.path.exists("ToG_{}.jsonl".format(file_name)):
        with open("ToG_{}.jsonl".format(file_name), "r") as outfile:
            entries = json.load(outfile)

    with open("ToG_{}.jsonl".format(file_name), "w") as outfile:
        entries.append(dict)
        json.dump(entries, outfile, indent=4)

    get_global_metrics().add_metrics(get_current_metrics())

    with open("ToG_{}_metrics.json".format(file_name), "w") as outfile:
        json.dump(get_global_metrics().to_dict(), outfile, indent=4)

    
def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    

def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False


def generate_without_explored_paths(question, args):
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return response


def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


def prepare_dataset(dataset_name):
    import globals
    global LABEL_PREDICATE
    global PREFIXES_LIST
    global PREFIXES
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        globals.LABEL_PREDICATE = globals.FREEBASE_LABEL_PREDICATE
        globals.PREFIXES_LIST = globals.FREEBASE_PREFIXES
        globals.PREFIXES = globals.get_prefixes(globals.FREEBASE_PREFIXES)
    elif dataset_name == 'cwq-bela':
        with open('../data/bela_cwq_topic_entities.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        globals.LABEL_PREDICATE = globals.FREEBASE_LABEL_PREDICATE
        globals.PREFIXES_LIST = globals.FREEBASE_PREFIXES
        globals.PREFIXES = globals.get_prefixes(globals.FREEBASE_PREFIXES)
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
        globals.LABEL_PREDICATE = globals.FREEBASE_LABEL_PREDICATE
        globals.PREFIXES_LIST = globals.FREEBASE_PREFIXES
        globals.PREFIXES = globals.get_prefixes(globals.FREEBASE_PREFIXES)
    elif dataset_name == 'webqsp-bela':
        with open('../data/bela_webqsp_topic_entities.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        globals.LABEL_PREDICATE = globals.FREEBASE_LABEL_PREDICATE
        globals.PREFIXES_LIST = globals.FREEBASE_PREFIXES
        globals.PREFIXES = globals.get_prefixes(globals.FREEBASE_PREFIXES)
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        globals.LABEL_PREDICATE = globals.FREEBASE_LABEL_PREDICATE
        globals.PREFIXES_LIST = globals.FREEBASE_PREFIXES
        globals.PREFIXES = globals.get_prefixes(globals.FREEBASE_PREFIXES)
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
    elif dataset_name == 'qald-9':
        with open('../data/qald_9_test_topic_entities.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'
        globals.LABEL_PREDICATE = globals.DBPEDIA_LABEL_PREDICATE
        globals.PREFIXES_LIST = globals.DBPEDIA_PREFIXES
        globals.PREFIXES = globals.get_prefixes(globals.DBPEDIA_PREFIXES)     
    elif dataset_name == 'qald-9-bela':
        with open('../data/bela_qald9_topic_entities.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'
        globals.LABEL_PREDICATE = globals.DBPEDIA_LABEL_PREDICATE
        globals.PREFIXES_LIST = globals.DBPEDIA_PREFIXES
        globals.PREFIXES = globals.get_prefixes(globals.DBPEDIA_PREFIXES)   
    elif dataset_name == 'lc-quad-1':
        with open('../data/lc_quad_1_test-data_topic_entities.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'corrected_question'  
        globals.LABEL_PREDICATE = globals.DBPEDIA_LABEL_PREDICATE
        globals.PREFIXES_LIST = globals.DBPEDIA_PREFIXES
        globals.PREFIXES = globals.get_prefixes(globals.DBPEDIA_PREFIXES)
    elif dataset_name == "lc-quad-1-bela":
        with open('../data/bela_lc_quad_topic_entities.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'  
        globals.LABEL_PREDICATE = globals.DBPEDIA_LABEL_PREDICATE
        globals.PREFIXES_LIST = globals.DBPEDIA_PREFIXES
        globals.PREFIXES = globals.get_prefixes(globals.DBPEDIA_PREFIXES)
    elif dataset_name == 'qald-10':
        with open('../data/qald_10_topic_entities.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'
        globals.LABEL_PREDICATE = globals.WIKIDATA_LABEL_PREDICATE
        globals.PREFIXES_LIST = globals.WIKIDATA_PREFIXES
        globals.PREFIXES = globals.get_prefixes(globals.WIKIDATA_PREFIXES)
    elif dataset_name == 'qald-10-bela':
        with open('../data/bela_qald10_topic_entities.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'
        globals.LABEL_PREDICATE = globals.WIKIDATA_LABEL_PREDICATE
        globals.PREFIXES_LIST = globals.WIKIDATA_PREFIXES
        globals.PREFIXES = globals.get_prefixes(globals.WIKIDATA_PREFIXES)
    elif dataset_name == 'lc-quad-2':
        with open('../data/lc_quad_2_test_topic_entities.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'
        globals.LABEL_PREDICATE = globals.WIKIDATA_LABEL_PREDICATE
        globals.PREFIXES_LIST = globals.WIKIDATA_PREFIXES
        globals.PREFIXES = globals.get_prefixes(globals.WIKIDATA_PREFIXES)
    elif dataset_name == 'spinach':
        with open('../data/spinach_test_topic_entities.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'
        globals.LABEL_PREDICATE = globals.WIKIDATA_LABEL_PREDICATE
        globals.PREFIXES_LIST = globals.WIKIDATA_PREFIXES
        globals.PREFIXES = globals.get_prefixes(globals.WIKIDATA_PREFIXES)
    elif dataset_name == 'bestiary':
        with open('../data/bestiary_topic_entities.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'
        globals.LABEL_PREDICATE = globals.BESTIARY_LABEL_PREDICATE
        globals.PREFIXES_LIST = globals.BESTIARY_PREFIXES
        globals.PREFIXES = globals.get_prefixes(globals.BESTIARY_PREFIXES)
    elif dataset_name == 'geoq1089':
        with open('../data/GeoQuestions1089_topic_entities.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'
        globals.LABEL_PREDICATE = globals.YAGO2GEO_LABEL_PREDICATE
        globals.PREFIXES_LIST = globals.YAGO2GEO_PREFIXES
        globals.PREFIXES = globals.get_prefixes(globals.YAGO2GEO_PREFIXES)
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string