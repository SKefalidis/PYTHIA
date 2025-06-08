import re
import time

from pathlib import Path
from platformdirs import user_config_dir
from typing import List
from src.elelem.provider import Provider
from src.evaluation.evaluator import Evaluatable
from src.engine.qa.kg_explorer.path_extractor import GroundedPath
from src.engine.qa.query_generator.query_db import QueryDb
from src.engine.qa.query_generator.query_generator_prompts import PROMPT_QUERY_GENERATION_ZEROSHOT, PROMPT_QUERY_GENERATION_FEWSHOT, PROMPT_GEOSPATIAL_PROMPT
from src.utils import get_kgaqa_tracker, llm_call
from src.logging import log, LogLevel, LogComponent, LogType
from src.engine.config import CONFIG


class QueryGenerator(Evaluatable):
    def __init__(self, model_id: Provider, query_db = None):
        self.model_id = model_id
        self.query_db: QueryDb = query_db
        
    def get_name(self):
        return "QueryGenerator"
    
    def get_resource(self):
        return ""

    def extract_sparql(self, text):
        pattern = r"```sparql(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        
        return text
    
    def predict(self, question: str, connections: List[str], grounded_paths: List[GroundedPath], failed_generations, entities, classes, geospatial_relations, force_zeroshot = False):
        log("Generating SPARQL query...", LogComponent.QUERY_GENERATOR, LogLevel.APPLICATION)
        log(f" - request: '{question}'", LogComponent.QUERY_GENERATOR, LogLevel.APPLICATION)
        log(f" - classes: {classes}", LogComponent.QUERY_GENERATOR, LogLevel.APPLICATION)
        log(f" - entities: {entities}", LogComponent.QUERY_GENERATOR, LogLevel.APPLICATION)
        log(f" - exploration plan: {connections}", LogComponent.QUERY_GENERATOR, LogLevel.APPLICATION)
        log(f" - geospatial relations: {geospatial_relations}", LogComponent.QUERY_GENERATOR, LogLevel.APPLICATION)
        paths_pretty = "\n".join(f"   - {p}" for p in grounded_paths) if grounded_paths else "   - (none)"
        log(f" - subgraph paths:\n{paths_pretty}", LogComponent.QUERY_GENERATOR, LogLevel.APPLICATION)
        # log(f" - grounded paths: {[path.get_formatted_information_string() for path in grounded_paths]}", LogComponent.QUERY_GENERATOR, LogLevel.APPLICATION)
        
        if (geospatial_relations is None or geospatial_relations == []) and (grounded_paths is None or grounded_paths == []):
            log("Warning: No geospatial relations and no grounded paths provided to the query generator.", LogComponent.QUERY_GENERATOR, LogLevel.WARNING)
            return "NO_QUERY"
        
        if force_zeroshot or self.query_db is None:
            return self._predict_zeroshot(question, connections, grounded_paths, failed_generations, entities, classes, geospatial_relations)
        else:
            return self._predict_icl(question, connections, grounded_paths, failed_generations, entities, classes, geospatial_relations)
        
    
    def _predict_zeroshot(self, question: str, connections: List[str], grounded_paths: List[GroundedPath], failed_generations, entities, classes, geospatial_relations):      
        if classes == None or not classes:
                classes = []
        if entities == None or not entities:
            entities = []
            
        # Clean up any None entries
        classes = [str(c) for c in classes if c is not None]
        entities = [str(e) for e in entities if e is not None]
        
        config_dir = Path(user_config_dir("pythia"))
        config_dir.mkdir(parents=True, exist_ok=True)
        user_prompt_path = config_dir / "user_instructions.txt"
        if 'wkt_uris' in CONFIG().get("kg_config"):
            geospatial_prompt = PROMPT_GEOSPATIAL_PROMPT.format(relations="\n".join(geospatial_relations),
                                                                wkt_access=CONFIG().get("kg_config")['wkt_uris'])
        else:
            geospatial_prompt = ""
    
        prompt = PROMPT_QUERY_GENERATION_ZEROSHOT.format(
            question=question,
            reasoning_path="\n".join(connections),
            triples = "\n\n".join([
                path.get_formatted_information_string()
                for path in grounded_paths
                if isinstance(path, GroundedPath)
            ]),
            entities=", ".join(entities),
            classes=", ".join(classes),
            user_instructions=user_prompt_path.read_text(encoding="utf-8") if user_prompt_path.exists() else "",
            geospatial=geospatial_prompt
        )
        log(f"[zeroshot] Prompt: {prompt}", LogComponent.QUERY_GENERATOR, LogLevel.DEBUG, LogType.PROMPT)
        
        get_kgaqa_tracker()._qg_prompt_query_gen_zero_shot_calls += 1
        start_time = time.time()
        
        generated = llm_call(self.model_id, prompt, max_tokens=768*2, temperature=0.5 * len(failed_generations))
        log(f"[zeroshot] Generated: {generated}", LogComponent.QUERY_GENERATOR, LogLevel.DEBUG, LogType.LLM_RESULT)
        
        get_kgaqa_tracker()._qg_prompt_query_gen_zero_shot_time += time.time() - start_time
        
        query = self.extract_sparql(generated)
        log(f"[zeroshot] Generated query: {query}", LogComponent.QUERY_GENERATOR, LogLevel.DEBUG, LogType.LLM_RESULT)
        
        return query

    def _predict_icl(self, question: str, connections: List[str], grounded_paths: List[GroundedPath], failed_generations, entities, classes, geospatial_relations):     
        if classes == None or not classes:
                classes = []
        if entities == None or not entities:
            entities = []
            
        # Clean up any None entries
        classes = [str(c) for c in classes if c is not None]
        entities = [str(e) for e in entities if e is not None]
         
        relevant_questions, relevant_queries = self.query_db.get_relevant_queries(question, top_k=3) 
        print(relevant_questions)
        print(relevant_queries)
        
        config_dir = Path(user_config_dir("pythia"))
        config_dir.mkdir(parents=True, exist_ok=True)
        user_prompt_path = config_dir / "user_instructions.txt"
        if 'wkt_uris' in CONFIG().get("kg_config"):
            geospatial_prompt = PROMPT_GEOSPATIAL_PROMPT.format(relations="\n".join(geospatial_relations),
                                                                wkt_access=CONFIG().get("kg_config")['wkt_uris'])
        else:
            geospatial_prompt = ""
        
        prompt = PROMPT_QUERY_GENERATION_FEWSHOT.format(
            question=question,
            reasoning_path="\n".join(connections),
            triples = "\n\n".join([
                path.get_formatted_information_string()
                for path in grounded_paths
                if isinstance(path, GroundedPath)
            ]),
            entities=", ".join(entities),
            classes=", ".join(classes),
            examples="\n\n".join([
                f"Question: {q}\nQuery: {r}" for q, r in zip(relevant_questions, relevant_queries)
            ]),
            user_instructions=user_prompt_path.read_text(encoding="utf-8") if user_prompt_path.exists() else "",
            geospatial=geospatial_prompt
            # failed_generations="\n".join([f"{gen[0]}: {gen[1]}" for gen in failed_generations] if failed_generations else [])
        )
        log(f"[ICL] Prompt: {prompt}", LogComponent.QUERY_GENERATOR, LogLevel.DEBUG, LogType.PROMPT)
        
        get_kgaqa_tracker()._qg_prompt_query_gen_icl_calls += 1
        start_time = time.time()
        
        generated = llm_call(self.model_id, prompt, max_tokens=768*2, temperature=0.5 * len(failed_generations))
        log(f"[ICL] Generated: {generated}", LogComponent.QUERY_GENERATOR, LogLevel.DEBUG, LogType.LLM_RESULT)
        
        get_kgaqa_tracker()._qg_prompt_query_gen_icl_time += time.time() - start_time
        
        query = self.extract_sparql(generated)
        log(f"[ICL] Generated query: {query}", LogComponent.QUERY_GENERATOR, LogLevel.DEBUG, LogType.LLM_RESULT)
        
        return query