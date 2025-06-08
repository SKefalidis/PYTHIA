from operator import is_
from site import PREFIXES
import traceback
from SPARQLWrapper import SPARQLWrapper, JSON
from utils import run_llm, compute_bm25_similarity, retrieve_top_docs, extract_answer, save_2_jsonl, if_true, clean_scores, all_unknown_entity, del_unknown_entity
from globals import get_current_metrics, get_global_metrics, get_password, get_user
import globals

SPARQLPATH = "None"
def set_sparql_path(sparql_path):
    global SPARQLPATH
    SPARQLPATH = sparql_path

# pre-defined sparqls
sparql_head_relations = """
{prefixes}
SELECT ?relation WHERE {{
    {entity} ?relation ?x .
}}
"""
sparql_tail_relations = """
{prefixes}
SELECT ?relation WHERE {{
    ?x ?relation {entity} .
}}"""

sparql_tail_entities_extract = """
{prefixes}
SELECT ?tailEntity WHERE {{
    {entity} {predicate} ?tailEntity .
}}"""
sparql_head_entities_extract = """
{prefixes}
SELECT ?tailEntity WHERE {{
    ?tailEntity {predicate} {entity} .
}}"""

sparql_id = """
{prefixes}
SELECT DISTINCT ?tailEntity WHERE {{
    {{
        {entity} <{label_predicate_uri}> ?tailEntity .
    }}
    UNION 
    {{
        {entity} <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .
    }}
}}"""

# FIXME: Will leave this as is.
def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)

# FIXME: Must find type-predicate and name-predicate for all KGs. It turns out that this is a speed optimization, therefore disabled by default.
def abandon_rels(relation, type_predicates=['type.object.type'], name_predicates=['type.object.name']):
    for predicate in type_predicates:
        if relation in predicate:
            return True
    for predicate in name_predicates:
        if relation in predicate:
            return True
    if "sameAs" in relation:
        return True
    # Disabled KG-specific optimizations.
    # if relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
    #     return True
    return False

def replace_relation_prefix(relations, prefixes):
    "Also prunes non-relation results."
    new_relations = []
    for relation in relations:
        relation_value = relation['relation']['value']
        for prefix in prefixes:
            if relation_value.startswith(prefix[1]):
                new_relations.append(relation_value.replace(prefix[1], prefix[0] + ":"))
                break
    return new_relations

def replace_entities_prefix(entities, prefixes):
    "Also prunes non-entity results."
    new_entities = []
    new_values = []
    for entity in entities:
        entity_value = entity['tailEntity']['value']
        is_value = True
        for prefix in prefixes:
            if entity_value.startswith(prefix[1]):
                new_entities.append(entity_value.replace(prefix[1], prefix[0] + ":"))
                is_value = False
                break
        if is_value:
            new_values.append(entity_value)
    return new_entities, new_values

def expand_prefix(entity):
    if ("http:" in entity or "https:" in entity) and entity.startswith("<") and entity.endswith(">"):
        return entity
    elif "http:" in entity or "https:" in entity:
        return "<" + entity + ">"
    for prefix in globals.PREFIXES_LIST:
        if entity.startswith(prefix[0] + ":"):
            return "<" + entity.replace(prefix[0] + ":", prefix[1]) + ">"
    return entity

def execurte_sparql(sparql_query):
    get_current_metrics().SPARQL_CALLS += 1
    start_time = time.time()
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setCredentials(get_user(), get_password())
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(91)
    print(sparql_query)
    try:
        results = sparql.query().convert()
    except Exception as e:
        print("SPARQL failed:", e)
        print(traceback.format_exc())
        results = {"results": {"bindings": []}}
    get_current_metrics().SPARQL_TIME += time.time() - start_time
    return results["results"]["bindings"]

def id2entity_name_or_type(entity_id):
    if isinstance(globals.LABEL_PREDICATE, str) and globals.LABEL_PREDICATE != "":
        get_current_metrics().SPARQL_CALLS += 1
        start_time = time.time()
        sparql_query = sparql_id.format(prefixes=globals.PREFIXES, entity=expand_prefix(entity_id), label_predicate_uri=globals.LABEL_PREDICATE)
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setCredentials(get_user(), get_password())
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        sparql.setTimeout(91)
        print(sparql_query)
        try:
            results = sparql.query().convert()
        except Exception as e:
            print("SPARQL failed:", e)
            print(traceback.format_exc())
            results = {"results": {"bindings": []}}
        get_current_metrics().SPARQL_TIME += time.time() - start_time
        if len(results["results"]["bindings"])==0:
            return "UnName_Entity"
        else:
            return results["results"]["bindings"][0]['tailEntity']['value']
    elif isinstance(globals.LABEL_PREDICATE, list) and len(globals.LABEL_PREDICATE) > 0:
        for label_predicate in globals.LABEL_PREDICATE:
            get_current_metrics().SPARQL_CALLS += 1
            start_time = time.time()
            sparql_query = sparql_id.format(prefixes=globals.PREFIXES, entity=expand_prefix(entity_id), label_predicate_uri=label_predicate)
            sparql = SPARQLWrapper(SPARQLPATH)
            sparql.setCredentials(get_user(), get_password())
            sparql.setQuery(sparql_query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(91)
            print(sparql_query)
            try:
                results = sparql.query().convert()
            except Exception as e:
                print("SPARQL failed:", e)
                print(traceback.format_exc())
                results = {"results": {"bindings": []}}
            get_current_metrics().SPARQL_TIME += time.time() - start_time
            if len(results["results"]["bindings"])>0:
                return results["results"]["bindings"][0]['tailEntity']['value']
        return "UnName_Entity"
    else:
        return re.split(r'[#/]', entity_id)[-1]
    
from freebase_func import *
from prompt_list import *
import json
import time
import openai
import re
from prompt_list import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer


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


def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    return extract_relation_prompt % (args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations) + "\nA: "
        

def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '


def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args):
    sparql_relations_extract_head = sparql_head_relations.format(prefixes=globals.PREFIXES, entity=expand_prefix(entity_id))
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations, globals.PREFIXES_LIST)
    head_relations = list(set(head_relations))
    
    sparql_relations_extract_tail= sparql_tail_relations.format(prefixes=globals.PREFIXES, entity=expand_prefix(entity_id))
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations, globals.PREFIXES_LIST)
    tail_relations = list(set(tail_relations))

    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations+tail_relations
    total_relations.sort()  # make sure the order in prompt is always equal

    print("Total relations before pruning: ", total_relations)
    
    if args.prune_tools == "llm":
        prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)
        print(prompt)
        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        print(result)
        flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations) 
        print("Retrieved relations with scores: ", retrieve_relations_with_scores)

    elif args.prune_tools == "bm25":
        topn_relations, topn_scores = compute_bm25_similarity(question, total_relations, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 
    else:
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        topn_relations, topn_scores = retrieve_top_docs(question, total_relations, model, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 

    if flag:
        # print("Retrieved relations with scores: ", retrieve_relations_with_scores)
        return retrieve_relations_with_scores
    else:
        return [] # format error or too small max_length
    
    
def entity_search(entity, relation, head=True):
    if head:
        tail_entities_extract = sparql_tail_entities_extract.format(prefixes=globals.PREFIXES, entity=expand_prefix(entity), predicate=expand_prefix(relation))
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract.format(prefixes=globals.PREFIXES, entity=expand_prefix(entity), predicate=expand_prefix(relation))
        entities = execurte_sparql(head_entities_extract)

    entity_ids, literal_values = replace_entities_prefix(entities, globals.PREFIXES_LIST) # Replacement also filters non-entities as the original code did (returned as separate lists).
    
    # print("Retrieved entities: ", entity_ids)
    
    return entity_ids, literal_values # literal values are only used in improved ToG


def entity_score(question, entity_candidates_id, score, relation, args):
    entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id]
    if all_unknown_entity(entity_candidates):
        return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    entity_candidates = del_unknown_entity(entity_candidates)
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id
    
    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)
    if args.prune_tools == "llm":
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)

        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        return [float(x) * score for x in clean_scores(result, entity_candidates)], entity_candidates, entity_candidates_id

    elif args.prune_tools == "bm25":
        topn_entities, topn_scores = compute_bm25_similarity(question, entity_candidates, args.width)
    else:
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        topn_entities, topn_scores = retrieve_top_docs(question, entity_candidates, model, args.width)
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    return [float(x) * score for x in topn_scores], topn_entities, entity_candidates_id


def improved_entity_score(question, entity_candidates_id, literal_values, score, relation, args):
    entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id]
    literal_candidates = literal_values

    combined_candidates = entity_candidates + literal_candidates
    combined_ids = entity_candidates_id + ["LITERAL_"+str(i) for i in range(len(literal_candidates))]

    if all_unknown_entity(combined_candidates):
        return [1/len(combined_candidates) * score] * len(combined_candidates), combined_candidates, combined_ids
    combined_candidates = del_unknown_entity(combined_candidates)
    if len(combined_candidates) == 1:
        return [score], combined_candidates, combined_ids
    if len(combined_candidates) == 0:
        return [0.0], combined_candidates, combined_ids
    
    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(combined_candidates, combined_ids))
    combined_candidates, combined_ids = zip(*zipped_lists)
    combined_candidates = list(combined_candidates)
    combined_ids = list(combined_ids)
    if args.prune_tools == "llm":
        prompt = construct_entity_score_prompt(question, relation, combined_candidates)

        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        return [float(x) * score for x in clean_scores(result, combined_candidates)], combined_candidates, combined_ids
    else:
        raise NotImplementedError("Improved entity scoring is only implemented for LLM pruning.")

    
def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


def half_stop(question, cluster_chain_of_entities, depth, args):
    print("No new knowledge added during search depth %d, stop searching." % depth)
    answer = generate_answer(question, cluster_chain_of_entities, args)
    save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset)


def generate_answer(question, cluster_chain_of_entities, args): 
    prompt = answer_prompt + question + '\n'
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return result


def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args):
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[:args.width], sorted_candidates[:args.width], sorted_topic_entities[:args.width], sorted_head[:args.width], sorted_scores[:args.width]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:
        return False, [], [], [], []
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

    tops = [id2entity_name_or_type(entity_id) for entity_id in tops]
    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads


def reasoning(question, cluster_chain_of_entities, args):
    prompt = prompt_evaluate + question
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    
    result = extract_answer(response)
    if if_true(result):
        return True, response
    else:
        return False, response