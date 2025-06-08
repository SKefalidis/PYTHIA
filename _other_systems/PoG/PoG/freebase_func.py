from SPARQLWrapper import SPARQLWrapper, JSON
from globals import *
import globals
import re

SPARQLPATH = "None"
def set_sparql_path(sparql_path):
    global SPARQLPATH
    SPARQLPATH = sparql_path

# pre-defined sparqls
sparql_head_relations = """
{prefixes}
SELECT ?relation WHERE {{
    {entity} ?relation ?x .
}}"""
sparql_tail_entities_and_relations = """
{prefixes}
SELECT ?relation ?tailEntity
WHERE {{
    {entity} ?relation ?tailEntity .
}}"""

sparql_tail_relations = """
{prefixes}
SELECT ?relation WHERE {{
    ?x ?relation {entity} .
}}"""
sparql_head_entities_and_relations = """
{prefixes}
SELECT ?relation ?headEntity
WHERE {{
    ?headEntity ?relation {entity} .
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

# FIXME: Must find type-predicate and name-predicate for all KGs.
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
    "Also prunes non-entity results." # TODO: Lacks Improved ToG's ability to handle literals. If you have the time, also try that.
    new_entities = []
    for entity in entities:
        entity_value = entity['tailEntity']['value']
        for prefix in prefixes:
            if entity_value.startswith(prefix[1]):
                new_entities.append(entity_value.replace(prefix[1], prefix[0] + ":"))
                break
    return new_entities

# TODO: See ToG for a more complete version of this function that supports other KGs.
# def expand_prefix(entity):
#     if "http:" in entity or "https:" in entity:
#         return "<" + entity + ">"
#     for prefix in globals.PREFIXES_LIST:
#         if entity.startswith(prefix[0] + ":"):
#             return "<" + entity.replace(prefix[0] + ":", prefix[1]) + ">"
#     return entity

def expand_prefix(entity):
    if ("http:" in entity or "https:" in entity) and entity.startswith("<") and entity.endswith(">"):
        return entity
    elif "http:" in entity or "https:" in entity:
        return "<" + entity + ">"
    for prefix in globals.PREFIXES_LIST:
        if entity.startswith(prefix[0] + ":"):
            return "<" + entity.replace(prefix[0] + ":", prefix[1]) + ">"
    return entity

import time

from threading import Lock

# Serialize outbound SPARQL calls to avoid overwhelming the endpoint.
_sparql_mutex = Lock()

def execurte_sparql(sparql_txt):
    return execute_sparql(sparql_txt)

def execute_sparql(sparql_txt):
    with _sparql_mutex:
        get_current_metrics().SPARQL_CALLS += 1
        start_time = time.time()
        get_current_metrics().SPARQL_CALLS += 1
        start_time = time.time()
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setCredentials(get_user(), get_password())
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        # print(sparql_txt)
        results = sparql.query().convert()
        # print("Results:", results["results"]["bindings"])
        get_current_metrics().SPARQL_TIME += time.time() - start_time
        return results["results"]["bindings"]
from functools import lru_cache
import re
@lru_cache(maxsize=1024)
def id2entity_name_or_type(entity_id):
    with _sparql_mutex:
        
        get_current_metrics().SPARQL_CALLS += 1
        
        if isinstance(globals.LABEL_PREDICATE, str) and globals.LABEL_PREDICATE != "":
            start_time = time.time()
            query = sparql_id.format(prefixes=globals.PREFIXES, entity=expand_prefix(entity_id), label_predicate_uri=globals.LABEL_PREDICATE)
            sparql = SPARQLWrapper(SPARQLPATH)
            sparql.setCredentials(get_user(), get_password())
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            # print(query)
            # results = sparql.query().convert()
            results = []
            attempts = 0
            while attempts < 3:  # Set the number of retries
                print(f"Attempt {attempts + 1} to execute SPARQL query.")
                print(query)
                try:
                    results = sparql.query().convert()
                    break
                    # return results["results"]["bindings"]
                except Exception as e:
                    print("404 Error encountered. Retrying after 2 seconds...")
                    print(sparql)
                    print(query)
                    time.sleep(2)  # Sleep for 2 seconds before retrying
                    attempts += 1  

            if attempts == 3:
                print("Failed to execute after multiple attempts.")
                
            get_current_metrics().SPARQL_TIME += time.time() - start_time
            
            if len(results["results"]["bindings"]) == 0:
                print("No results found for entity:", entity_id)
                return "Unnamed Entity"
            else:
                # print(results)
                # First, filter to find results with 'xml:lang': 'en'
                english_results = [result['tailEntity']['value'] for result in results["results"]["bindings"] if result['tailEntity'].get('xml:lang') == 'en']
                if english_results:
                    return english_results[0]  # Return the first English result

                # If no English results, find entries that match English letters or numbers
                alphanumeric_results = [result['tailEntity']['value'] for result in results["results"]["bindings"]
                                        if re.match("^[a-zA-Z0-9 ]+$", result['tailEntity']['value'])]
                if alphanumeric_results:
                    return alphanumeric_results[0]  # Return the first alphanumeric result

                return "Unnamed Entity"
        else:
            return re.split(r'[#/]', entity_id)[-1]

