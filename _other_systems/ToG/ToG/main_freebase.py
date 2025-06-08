from narwhals import lit
# from numpy.strings import lower
from tqdm import tqdm
import argparse
from utils import *
import random
import os
import json
from client import *
from globals import *

if __name__ == '__main__':
    from freebase_func import *

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=512, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=False, help="whether removing unnecessary relations.") # Set to FALSE to disable KG-specific optimizations
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-4.1-mini", help="base LLM model.") # Set the same LLM as Pythia
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    parser.add_argument("--endpoint", type=str)
    parser.add_argument("--user", type=str, default="user")
    parser.add_argument("--password", type=str, default="user")
    parser.add_argument("--improved_tog", action='store_true',
                        help="whether using improved ToG with support for literal values.")
    args = parser.parse_args()

    set_user(args.user)
    set_password(args.password)

    set_improved_tog(args.improved_tog)

    if args.improved_tog:
        print("Improved ToG enabled: literal values are considered during entity search and scoring.")
    else:
        print("Original ToG enabled: only entity candidates are considered during entity search and scoring.")

    datas, question_string = prepare_dataset(args.dataset)
    print("Start Running ToG on %s dataset." % args.dataset)
    
    entries_r_count = -1
    if os.path.exists("ToG_{}.jsonl".format(args.dataset)):
        with open("ToG_{}.jsonl".format(args.dataset), "r") as infile:
            entries = json.load(infile)
            entries_r_count = len(entries)

    entries_m_count = -1
    if os.path.exists("ToG_{}_metrics.json".format(args.dataset)):
        with open("ToG_{}_metrics.json".format(args.dataset), "r") as infile:
            get_global_metrics().load_from_file("ToG_{}_metrics.json".format(args.dataset))
            entries_m_count = get_global_metrics().QUESTIONS

    if entries_r_count > 0 and entries_m_count > 0 and entries_r_count != entries_m_count:
        print("Warning: The number of entries in ToG_{}.jsonl and ToG_{}_metrics.json are inconsistent.".format(args.dataset, args.dataset))
        exit(1)
    elif entries_r_count > 0 and entries_m_count > 0 and entries_r_count == entries_m_count:
        print("Resuming from entry %d." % entries_r_count)
        datas = datas[entries_r_count:]
    
    for data in tqdm(datas):
        get_current_metrics(new=True)
        set_timer_to_current()
        set_sparql_path(args.endpoint)

        question = data[question_string]
        # topic_entity = data['topic_entity']
        # FIXME: Temporary workaround for Freebase!!
        
        if args.dataset.lower() in ['webqsp', 'cwq']:
            topic_entity = {"ns:"+k: v for k, v in data['topic_entity'].items()}
            
        if args.dataset.lower() in ['qald-10', 'qald-9', 'lc-quad-1', 'lc-quad-2', 'spinach', 'bestiary', 'geoq1089', 'qald-10-bela', 'qald-9-bela', 'lc-quad-1-bela', 'lc-quad-2-bela', 'webqsp-bela', 'cwq-bela']:
            topic_entity = {k: v for k, v in data['topic_entities'].items()}
    
        cluster_chain_of_entities = []
        if len(topic_entity) == 0:
            results = generate_without_explored_paths(question, args)
            save_2_jsonl(question, results, [], file_name=args.dataset)
            continue
        pre_relations = []
        pre_heads= [-1] * len(topic_entity)
        flag_printed = False
        for depth in range(1, args.depth+1):
            current_entity_relations_list = []
            i=0
            for entity in topic_entity:
                if entity!="[FINISH_ID]" and len(pre_heads)>0:
                    try:
                        retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i], question, args)  # best entity triplet, entitiy_id
                        print("Retrieved relations for entity %s: %s" % (entity, retrieve_relations_with_scores))
                        current_entity_relations_list.extend(retrieve_relations_with_scores)
                    except Exception as e:
                        print(topic_entity)
                        print(pre_heads)
                        print(entity)
                        print(i)
                        raise e
                i+=1
            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities_id = []
            total_topic_entities = []
            total_head = []

            for entity in current_entity_relations_list:
                if entity['head']:
                    # literal values empty unless improved ToG is enabled
                    entity_candidates_id, literal_values = entity_search(entity['entity'], entity['relation'], True)
                else:
                    entity_candidates_id, literal_values = entity_search(entity['entity'], entity['relation'], False)

                # only entity candidates unless improved ToG is enabled
                combined = entity_candidates_id + literal_values
                
                if args.prune_tools == "llm":
                    if len(combined) >=20:
                        combined = random.sample(combined, args.num_retain_entity)
                    if args.improved_tog == False:
                        entity_candidates_id = combined
                        literal_values = []
                    else:
                        entity_candidates_id = [c for c in combined if c in entity_candidates_id]
                        literal_values = [c for c in combined if c in literal_values]

                if len(entity_candidates_id) == 0:
                    if args.improved_tog == False: # Original ToG behavior
                        continue
                    elif len(literal_values) == 0:
                        continue
                
                if args.improved_tog == False:
                    scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, literal_values, entity['score'], entity['relation'], args)
                else: # also consider literal values
                    scores, entity_candidates, entity_candidates_id = improved_entity_score(question, entity_candidates_id, literal_values, entity['score'], entity['relation'], args)
                print("Entity candidates after scoring: ", entity_candidates_id)
                
                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)
                print(total_candidates)
            
            if len(total_candidates) ==0:
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
                break
                
            flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args)
            cluster_chain_of_entities.append(chain_of_entities)
            if flag:
                stop, results = reasoning(question, cluster_chain_of_entities, args)
                if stop:
                    print("ToG stoped at depth %d." % depth)
                    save_2_jsonl(question, results, cluster_chain_of_entities, file_name=args.dataset)
                    flag_printed = True
                    break
                else:
                    print("depth %d still not find the answer." % depth)
                    flag_finish, entities_id = if_finish_list(entities_id)
                    if flag_finish:
                        half_stop(question, cluster_chain_of_entities, depth, args)
                        flag_printed = True
                    else:
                        topic_entity = {entity: id2entity_name_or_type(entity) for entity in entities_id if not entity.startswith("LITERAL")} # avoid literal values as topic entities
                        continue
            else:
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
        
        if not flag_printed:
            results = generate_without_explored_paths(question, args)
            save_2_jsonl(question, results, [], file_name=args.dataset)
