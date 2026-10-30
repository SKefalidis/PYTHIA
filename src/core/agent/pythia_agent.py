from __future__ import annotations
import os
import yaml
import json
import time
import argparse

from typing import Any, Dict, List, Optional
from collections import OrderedDict

import litellm
from litellm import completion, CustomStreamWrapper, ModelResponse, Usage

from src.dataloaders.dataset import DatasetFactory
from src.config import AGENT_SETTINGS, GENERAL_SETTINGS, GRAPH_CONFIGURATIONS, LLM_SETTINGS, QUERY_DATABASE_SETTINGS, RUNTIME_SETTINGS, PythiaArgParser
from src.query_index.query_db import QueryDb
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.utils_gost import extract_uris
from src.core.tools import *
from src.metrics import get_kgaqa_tracker, snapshot_metrics, change_since_snapshot

from src.core.agent.pythia_agent_prompts import *

# litellm.modify_params = True
litellm.drop_params = True

class PythiaAgent:

    def __init__(self, 
                 kg: KnowledgeGraphs, 
                 tools: List[AvailableTools] = [], 
                 db_dataset_name: str = "", 
                 db_dataset_path: str = ""):
        # Model setup
        self.model: str = LLM_SETTINGS.get_litellm_model_endpoint()  # e.g., "openai/gpt-4.1-nano"

        # Setup knowledge graph and its config
        self.kg = kg
            
        if db_dataset_path:
            dataset = DatasetFactory.create_dataset(db_dataset_name, db_dataset_path)  # Ensure dataset is loaded
            self._query_db = QueryDb(dataset)
            self.use_examples = True
        else:
            self._query_db = None
            self.use_examples = False
        
        if not tools:
            if not AGENT_SETTINGS.tools.no_find:
                tools.append(AvailableTools.FIND_ANCHORS)
            else:
                print("WARNING: Running without the find tool, which may lead to significantly worse performance.")
            if not AGENT_SETTINGS.tools.no_graph_search:
                tools.append(AvailableTools.GRAPH_SEARCH)
            else:
                print("WARNING: Running without the graph search tool, which may lead to significantly worse performance.")
            if not AGENT_SETTINGS.tools.no_stepwise_search:
                tools.append(AvailableTools.STEPWISE_SEARCH)
            else:
                print("WARNING: Running without the stepwise search tool, which may lead to significantly worse performance.")
            if not AGENT_SETTINGS.tools.no_get_predicates:
                tools.append(AvailableTools.GET_PREDICATES)
            else:
                print("WARNING: Running without the get predicates tool, which may lead to significantly worse performance.")

        # Tools
        self.available_tools: OrderedDict[AvailableTools, Tool] = OrderedDict()
        if AvailableTools.FIND_ANCHORS in tools:
            self.finder = Finder(kg) # Instances of resource heavy classes are managed by the Agent class and shared across tools    
            self.find_anchors_tool = FindAnchorsTool(self.finder, self.kg, k=10)
            self.available_tools[AvailableTools.FIND_ANCHORS] = self.find_anchors_tool
        else:
            self.find_anchors_tool = None

        if AvailableTools.STEPWISE_SEARCH in tools:
            self.stepwise_search_tool = StepwiseSearchTool(self.kg)
            self.available_tools[AvailableTools.STEPWISE_SEARCH] = self.stepwise_search_tool
        else:
            self.stepwise_search_tool = None
            
        if AvailableTools.GRAPH_SEARCH in tools:
            self.graph_search_tool = GraphSearchTool(self.kg)
            self.available_tools[AvailableTools.GRAPH_SEARCH] = self.graph_search_tool
        else:
            self.graph_search_tool = None
            
        if AvailableTools.GET_PREDICATES in tools:
            self.get_predicates_tool = PredicatesTool(self.kg)
            self.available_tools[AvailableTools.GET_PREDICATES] = self.get_predicates_tool
        else:
            self.get_predicates_tool = None
            
        enable_explanation_of_empty_results = not AGENT_SETTINGS.tools.no_empty_query_investigation
        self.query_execution_tool = QueryExecutionTool(self.kg, enable_explanation_of_empty_results=enable_explanation_of_empty_results)
            
        # Agent settings (can be surfaced in CONFIG if desired)
        self.max_steps: int = AGENT_SETTINGS.settings.max_steps
        self.max_tokens: int = AGENT_SETTINGS.settings.max_tokens
        if "gpt-5" in self.model:
            self.max_tokens = self.max_tokens * 3  # GPT-5 models need more tokens to support reasoning.
        self.automate_last_step: bool = AGENT_SETTINGS.settings.automate_last_step
        self.grasp_prompt: bool = AGENT_SETTINGS.prompt.grasp_prompt  # whether to use the relational model (graph search + stepwise search) or just rely on get_predicates_for_node for relation discovery. This is mostly for testing and ablations.
        self.basic_prompt: bool = AGENT_SETTINGS.prompt.basic_prompt
        self.basic_relational_prompt: bool = AGENT_SETTINGS.prompt.basic_relational_prompt
        self.no_examples_prompt: bool = AGENT_SETTINGS.prompt.no_examples_prompt
        
        # print(f"GRASP prompt enabled: {self.grasp_prompt}")
        # print(f"Basic prompt enabled: {self.basic_prompt}")
        # print(f"Basic relational prompt enabled: {self.basic_relational_prompt}")
        # print(f"Basic no examples prompt enabled: {self.no_examples_prompt}")
        
        # State
        self.messages = []
        self.remaining_steps = self.max_steps
        self.last_query = ""
        
        self.executed_query = False
        
        print(f"PYTHIA Agent initialized with:")
        print(f"- Knowledge Graph: {self.kg.name}")
        print(f"- Available Tools: {list(self.available_tools.keys())}")
        print(f"- Model: {self.model}")
        print(f"- Max Steps: {self.max_steps}")
        print(f"- Max Tokens: {self.max_tokens}")
        print(f"- Few-shot examples enabled: {self.use_examples}")

    # --------------------------- Public API ---------------------------
    def answer(self, question: str, stopwatch: bool = False, sparql_only: bool = True, gold_topic_entity_uris: List[str] = [], bela_topic_entity_uris: List[str] = []) -> str:
        if self.find_anchors_tool:
            self.find_anchors_tool.returned_entities = set()
            self.find_anchors_tool.returned_classes = set()
        
        if self.graph_search_tool:
            self.graph_search_tool.start_end_given = []
            self.graph_search_tool.tuples_returned = []
        
        if self.stepwise_search_tool:
            self.stepwise_search_tool.start_given = []
        
        from pprint import pprint
        
        snapshot = snapshot_metrics(get_kgaqa_tracker())
        
        start_total = time.time()
        self.last_query = ""

        # System instruction for the assistant role. We rely on tool calling for structure.
        system_content = self._build_system_prompt(question)
        
        user_content = self._build_user_prompt(question)
        
        if gold_topic_entity_uris is not None and len(gold_topic_entity_uris) > 0:
            topic_entities_message = self._build_topic_entities_prompt(gold_topic_entity_uris, [])
        elif bela_topic_entity_uris is not None and len(bela_topic_entity_uris) > 0:
            topic_entities_message = self._build_topic_entities_prompt([], bela_topic_entity_uris)
        else:
            topic_entities_message = ""
    
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content + topic_entities_message},
        ]
        
        if self.find_anchors_tool == None:
            messages.append({
                "role": "developer" if "gpt" in self.model else "user",
                "content": f"Note: The `{AvailableTools.FIND_ANCHORS.value}` tool is disabled, so you won't be able to retrieve entities or classes directly grounded in the question. You will have to rely on the other tools to find any necessary entities, classes, properties, and relations. This may make it more difficult to answer the question, especially if it requires specific entities or classes mentioned in the question."
            })
            
        if self.stepwise_search_tool == None:
            messages.append({
                "role": "developer" if "gpt" in self.model else "user",
                "content": f"Note: The `{AvailableTools.STEPWISE_SEARCH.value}` tool is disabled, so you won't be able to perform a stepwise search (beam search) to find paths between nodes."
            })
            
        if self.graph_search_tool == None:
            messages.append({
                "role": "developer" if "gpt" in self.model else "user",
                "content": f"Note: The `{AvailableTools.GRAPH_SEARCH.value}` tool is disabled, so you won't be able to perform a graph search (bidirectional BFS) to find connections between two nodes."
            })

        # Build tool schemas based on the enabled tool names. If None -> all tools. If [] -> no tools.
        tools = self._tool_schemas()
        tools_in_use = [name['function']['name'] for name in tools]
        tool_calls_count = {name: 0 for name in tools_in_use}
        tool_calls_time = {name: 0.0 for name in tools_in_use}
        tool_results: Dict[str, Any] = {}
        final_sparql: Optional[str] = None

        answer = False
        previous_input = 0
        for steps in range(self.max_steps):
            start_time = time.time()
            
            # Skip last step, it can only be used to output the final query
            if self.automate_last_step and steps == self.max_steps -1:
                if self.last_query != "":
                    final_sparql = self.last_query
                    answer = True
                    break
            
            # reminder = [{
            #     "role": "developer",
            #     "content": f"Explain your next step immediately before executing a tool call. Be concise and precise if everything is going according to plan. If a re-plan is needed be more elaborate. You must always take one or more of the actions: `retrieve_entities_and_classes`, `get_predicates_for_node`, `bidirectional_bfs`, `beam_search`, `execute_query`, `use_last_executed_query_as_answer`.",
            # }]
            # response = self._chat(messages + reminder, tools=tools)
            
            response = self._chat(messages, tools=tools)
            
            usage: Usage = response.usage # type: ignore
            get_kgaqa_tracker()._llm_calls += 1
            get_kgaqa_tracker()._llm_time += time.time() - start_time
            get_kgaqa_tracker()._llm_inputs += usage.prompt_tokens
            get_kgaqa_tracker()._llm_outputs += usage.completion_tokens
            get_kgaqa_tracker()._llm_tokens += usage.total_tokens
            
            choice = response.choices[0]
            msg = choice.message # type: ignore
            
            print("\n=== Agent Step", steps + 1, "===")
            print(f"LLM total input tokens: {usage.prompt_tokens}, new input tokens: {usage.prompt_tokens - previous_input}, output tokens: {usage.completion_tokens}, total tokens: {usage.total_tokens}")
            print()
            
            print(msg.content or "")
            
            if msg.content and msg.content.strip() != "":
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                })
            
            previous_input = usage.prompt_tokens

            # If assistant produced a tool call, execute it and append tool result
            if msg.tool_calls:
                for call in msg.tool_calls:
                    func = call.function
                    name = func.name
                    if not isinstance(name, str):
                        raise ValueError(f"Tool name must be a string. Got {name} of type {type(name)}")
                    
                    if name not in tools_in_use:
                        print(f"Model attempted to call tool '{name}' which is not in the list of available tools. Ignoring this tool call.")
                        continue

                    try:
                        args = json.loads(func.arguments) if isinstance(func.arguments, str) else (func.arguments or {})
                        args['question'] = question
                    except Exception:
                        args = {"_raw": func.arguments}
                        
                    print(f"Executing tool: {name} with args: {args}")

                    tool_start_time = time.time()
                    result = self._execute_tool(name, args)
                    tool_calls_count[name] += 1
                    tool_calls_time[name] += time.time() - tool_start_time
                    
                    tool_results[call.id] = result

                    if name == "use_last_executed_query_as_answer":
                        if self.last_query:
                            final_sparql = self.last_query
                            answer = True
                            break
                        else:
                            if "gpt-4" in self.model or "gpt-5" in self.model:
                                messages.append({
                                    "role": "developer",
                                    "content": f"You called `use_last_executed_query_as_answer`, but no query has been executed successfully yet, so there is no query to use as the answer. Please review the tool calling guidelines and make sure to only call `use_last_executed_query_as_answer` after a successful `execute_query` call that returned a valid SPARQL query."
                                })
                            else:
                                messages.append({
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "id": call.id,
                                            "type": "function",
                                            "function": {"name": name, "arguments": json.dumps(args)},
                                        }
                                    ],
                                    "content": None,
                                })
                                
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": call.id,
                                    "name": name,
                                    "content": f"You called `use_last_executed_query_as_answer`, but no query has been executed successfully yet, so there is no query to use as the answer. Please review the tool calling guidelines and make sure to only call `use_last_executed_query_as_answer` after a successful `execute_query` call that returned a valid SPARQL query."
                                })
                            break # let the model retry
                        
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)},
                            }
                        ],
                        "content": None,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    
                    if self.max_steps - (steps + 1) == 3:
                        if "gpt-4" in self.model or "gpt-5" in self.model:
                            messages.append({
                                "role": "developer",
                                "content": f"I want you to generate an answer in the next 2 steps.",
                            })
                    
                    pprint(result)   
            # else:
            #     messages.append({
            #         "role": "assistant",
            #         "content": msg.content or "",
            #     })      
            #     print("Assistant message without tool call:")
            #     print(msg.content or "")       
            
            if answer:
                break

        # if final_sparql is None:
        #     # One last try: ask to finalize
        #     messages.append({"role": "system", "content": "Please finalize by returning only the SPARQL query text."})
        #     response = self._chat(messages, tools=tools, tool_choice="none")
        #     content = response.choices[0].message.content or ""
        #     final_sparql = content
            
        # pprint(messages)

        # Normalize URIs like Pythia
        sparql_query = final_sparql
        
        if sparql_query is None or sparql_query.strip() == "":
            sparql_query = self.last_query

        if stopwatch:
            print(f"Agent total time: {time.time() - start_total:.2f} seconds in {steps} steps.") # type: ignore

        if sparql_only:
            return sparql_query
        else:
            used_classes = []
            used_entities = []
            if self.stepwise_search_tool:
                for uri in self.stepwise_search_tool.start_given:
                    if self.find_anchors_tool:
                        if uri in self.find_anchors_tool.returned_entities:
                            used_entities.append(uri)
                        if uri in self.find_anchors_tool.returned_classes:
                            used_classes.append(uri)
            if self.graph_search_tool:
                for start, end in self.graph_search_tool.start_end_given:
                    if self.find_anchors_tool:
                        if start in self.find_anchors_tool.returned_entities:
                            used_entities.append(start)
                        if end in self.find_anchors_tool.returned_entities:
                            used_entities.append(end)
                        if start in self.find_anchors_tool.returned_classes:
                            used_classes.append(start)
                        if end in self.find_anchors_tool.returned_classes:
                            used_classes.append(end)
            used_entities = set(used_entities)
            used_classes = set(used_classes)
            
            metrics = change_since_snapshot(snapshot)
            return {
                "sparql": sparql_query,
                "steps": steps,
                "elapsed": time.time() - start_total,
                "metrics": {
                    "SPARQL_CALLS": metrics.sparql_calls,
                    "SPARQL_TIME": metrics.sparql_time,
                    "LLM_CALLS": metrics.llm_calls,
                    "LLM_TIME": metrics.llm_time,
                    "LLM_INPUTS": metrics.llm_inputs,
                    "LLM_OUTPUTS": metrics.llm_outputs
                },
                "tool_calls_count": tool_calls_count,
                "tool_calls_time": tool_calls_time,
                "messages": messages,
                "used_entities": list(used_entities),
                "used_classes": list(used_classes),
                "found_entities": list(self.find_anchors_tool.returned_entities) if self.find_anchors_tool else [],
                "found_classes": list(self.find_anchors_tool.returned_classes) if self.find_anchors_tool else [],
                "graph_search_start_end": self.graph_search_tool.start_end_given if self.graph_search_tool else [],
                "graph_search_tuples_returned": self.graph_search_tool.tuples_returned if self.graph_search_tool else [],
                "empty_due_to_filters_count": self.query_execution_tool.empty_due_to_filters_count,
                "empty_due_to_invalid_triples_count": self.query_execution_tool.empty_due_to_invalid_triples_count,
                "empty_due_to_invalid_combination_count": self.query_execution_tool.empty_due_to_invalid_combination_count,
                "empty_due_to_select_vars_count": self.query_execution_tool.empty_due_to_select_vars_count,
                "empty_unknown_count": self.query_execution_tool.empty_unknown_count
            }
            
    def get_tools_str(self) -> str:
        tools_str = ''
        if AvailableTools.FIND_ANCHORS in self.available_tools:
            tools_str += "f"
        if AvailableTools.GRAPH_SEARCH in self.available_tools:
            tools_str += "g"
        if AvailableTools.STEPWISE_SEARCH in self.available_tools:
            tools_str += "s"
        if AvailableTools.GET_PREDICATES in self.available_tools:
            tools_str += "p"
        if tools_str == 'fgsp':
            tools_str = 'full'
        return tools_str

    def get_prompt_str(self) -> str:
        if self.basic_prompt:
            return "basic"
        if self.grasp_prompt:
            return "grasp"
        if self.basic_relational_prompt:
            return "basicr"
        if self.no_examples_prompt:
            return "noexamples"
        if self.use_examples:
            return "fewshot"
        else:
            return "zeroshot"

    # ------------------------- Internal utils ------------------------
    def _build_system_prompt(self, question: str) -> str:
        prefixes_text = self.kg.prefixes
        extra_prefixes = "\n" + prefixes_text if prefixes_text else ""
        
        if self.basic_prompt == True:
            return BASIC_PROMPT.format(KG=self.kg.name, PREFIXES=extra_prefixes).strip()
        
        if self.grasp_prompt == True:
            return GRASP_PROMPT.format(PREFIXES=extra_prefixes).strip()

        if self.no_examples_prompt == True:
            return NO_EXAMPLES_PROMPT.format(KG=self.kg.name, PREFIXES=extra_prefixes).strip()
            
        if self.basic_relational_prompt == True:
            return BASIC_RELATIONAL_PROMPT.format(KG=self.kg.name, PREFIXES=extra_prefixes).strip()

        if self.use_examples == False:
            return ZEROSHOT_PROMPT.format(KG=self.kg.name, PREFIXES=extra_prefixes).strip()
        else:
            if self._query_db is None:
                raise ValueError("Query DB is not initialized, but use_examples is True.")
            example_questions, example_queries = self._query_db.get_relevant_queries(question, top_k=3)
            
            examples_string = "\n" + '\n'.join([f"Question: {question}\nSPARQL Query:\n```{query}```" for question, query in zip(example_questions, example_queries)]) + "\n"
            used_uris_string = ""
            for query in example_queries:
                # print(query)
                uris = extract_uris(self.kg.prefixes + query)
                if uris is None:
                    continue
                for uri in uris:
                    label = self.kg.get_label_from_uri(uri)
                    used_uris_string += f"- {label} (<{uri}>)\n"
            
            return FEWSHOT_PROMPT.format(KG=self.kg.name, PREFIXES=extra_prefixes, EXAMPLES=examples_string, USED_URIS=used_uris_string).strip()
        
    def _build_user_prompt(self, question: str) -> str:
        ask_message = ""
        if question.lower().startswith("Does") or question.lower().startswith("Is") or question.lower().startswith("Are") or question.lower().startswith("Was") or question.lower().startswith("Were") or question.lower().startswith("Has") or question.lower().startswith("Have"):
            ask_message += "Remember that if this is a Yes/No question, you MUST produce an ASK SPARQL query as the final answer."
        prompt = f"{question} {ask_message}. Do not use \"SERVICE wikibase:label\" or similar services. They are not supported by the target endpoint. You do not need to return labels in your answers, URIs are sufficient."
        return prompt
    
    def _build_topic_entities_prompt(self, gold_topic_entity_uris: List[str], bela_topic_entity_uris: List[str]) -> str:
        if gold_topic_entity_uris == [] and bela_topic_entity_uris == []:
            return ""
        
        if gold_topic_entity_uris != [] and bela_topic_entity_uris != []:
            print("Please provide only one of gold_topic_entity_uris or bela_topic_entity_uris.")
            exit(1)
        
        prompt = ""
        if gold_topic_entity_uris:
            prompt = "The following entities can be used to answer my question. They are taken from the target knowledge graph:\n"
            if len(gold_topic_entity_uris) == 0:
                return ""
            entities_string = ""
            for uri in gold_topic_entity_uris:
                kgc = self.kg.get_kg_component(uri)
                if kgc is not None:
                    kg_string = f"{uri}: {kgc.label}, {kgc.description[:100]}, Types: {', '.join(kgc.parent_classes)}"
                else:
                    kg_string = f"{uri}: " + self.kg.get_label_from_uri(uri)
                entities_string += f"  - {kg_string}\n"
            prompt += entities_string
        if bela_topic_entity_uris:
            prompt = "Here are some relevant nodes (named entities and classes) from the target knowledge graph. You can use this information to help you generate the SPARQL query.\n"
            if len(bela_topic_entity_uris) == 0:
                return ""
            entities_string = ""
            for uri in bela_topic_entity_uris:
                if uri is None or uri.strip() == "":
                    continue
                kgc = self.kg.get_kg_component(uri)
                if kgc is not None:
                    kg_string = f"{uri}: {kgc.label}, {kgc.description[:100]}, Types: {', '.join(kgc.parent_classes)}"
                else:
                    kg_string = f"{uri}: " + self.kg.get_label_from_uri(uri)
                entities_string += f"  - {kg_string}\n"
            prompt += entities_string
        return prompt

    def _chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, tool_choice: str | dict | None = "auto"):
        """Chat using OpenAI APIs directly when available; otherwise, raise NotImplementedError.
        """
        if "gpt-4" not in self.model and "gpt-5" not in self.model:
            if messages[-1]["role"] != "user" and messages[-1]["role"] != "tool":
                messages.append({
                    "role": "user",
                    "content": "Please continue trying to answer the question using the provided tools.",
                })
        tries = 0
        while tries < 3:
            try:
                if "gpt-5.4" in self.model:
                    response: ModelResponse | CustomStreamWrapper = completion(self.model,
                                                                            messages = messages,
                                                                            max_tokens = self.max_tokens,
                                                                            tools = tools,
                                                                            tool_choice = tool_choice,
                                                                            temperature=0.3)
                elif "gpt-4" in self.model or "gpt-5" in self.model:
                    response: ModelResponse | CustomStreamWrapper = completion(self.model,
                                                                            messages = messages,
                                                                            max_tokens = self.max_tokens,
                                                                            tools = tools,
                                                                            tool_choice = tool_choice,
                                                                            temperature=0.3,
                                                                            reasoning_effort="low",
                                                                            prompt_cache_key = "agent_with_tools_cache")
                else:
                    response: ModelResponse | CustomStreamWrapper = completion(self.model,
                                                                            messages = messages,
                                                                            max_tokens = self.max_tokens,
                                                                            tools = tools,
                                                                            tool_choice = tool_choice,
                                                                            temperature=0.3)
                if isinstance(response, CustomStreamWrapper):
                    raise Exception("Streaming responses are not supported in PythiaAgent.")
                return response
            except Exception as e:
                print("Error during chat completion:", str(e))
                tries += 1
        print("Failed to get response from OpenAI after 3 tries. Messages were:")
        for msg in messages:
            print(f"  {msg['role']}: {msg['content']}")
        raise Exception("Failed to get response from OpenAI after 3 tries.")

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        print("=== Agent Tool Call ===")
        print(f"Tool: {name}, Args: {args}")
        start_time = time.time()
        
        results = None
        
        for _, tool in self.available_tools.items():
            if tool.name() == name:
                self.executed_query = False
                results = tool.function(**args)
                break
            
        if name == self.query_execution_tool.name():
            avoid_remodifying = False
            success, query, results = self.query_execution_tool.function(**args)
            if success:
                if self.executed_query:
                    # if "The query executed successfully but returned no results" in results:
                    avoid_remodifying = True
                if "The query executed successfully and returned" in results:
                    self.last_query = query
                self.executed_query = True
            return results
            # if avoid_remodifying:
            #     return results + " Time to change strategy. Find additional information in the knowledge graph using the available tools before executing another query."
            # else:
            #     return results
                
        if name == "use_last_executed_query_as_answer":
            return "ANSWER_USING_LAST_QUERY"
        
        if results is not None:
            print(f"Tool execution time for {name}: {time.time() - start_time:.2f} seconds.")
            return results
        
        return {"error": f"Unknown tool: {name}"}

    def _tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas"""
        schemas: Dict[str, Dict[str, Any]] = {tool.name(): tool.schema() for tool in self.available_tools.values()}
        schemas[self.query_execution_tool.name()] = self.query_execution_tool.schema()
        schemas["use_last_executed_query_as_answer"] = {
            "type": "function",
            "function": {
                "name": "use_last_executed_query_as_answer",
                "description": "Use the last executed SPARQL query as the final answer to the user's question. Use this after executing a query that you consider final and correct, using the 'execute_query' tool.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
        return schemas.values()
    
    # ------------------------- Arguments ------------------------
    @staticmethod
    def fill_parse_args(parser: argparse.ArgumentParser, graph_required: bool) -> argparse.ArgumentParser:        
        # Knowledge graph and endpoint
        graph_subgroup = parser.add_argument_group("PYTHIA - Target Knowledge Graph and Endpoint Settings (Required)")
        if graph_required:
            graph_subgroup.add_argument("--graph", type=str, required=graph_required, 
                                        choices=[key for key in GRAPH_CONFIGURATIONS.keys()],
                                        help="Target knowledge graph. \
                                            The settings below can be used to override the defaults for the chosen graph.")
        RUNTIME_SETTINGS.graph_fill_parse_args(graph_subgroup)
        RUNTIME_SETTINGS.endpoint_fill_parse_args(graph_subgroup)
        
        # Agent settings
        AGENT_SETTINGS.fill_parse_args(parser)
        
        # Dynamic Few-Shot Learning (DFSL) settings for question-query examples
        dfsl_subgroup = QUERY_DATABASE_SETTINGS.fill_parse_args(parser)
        dfsl_subgroup.add_argument("--examples_dataset", type=str, help="Examples dataset name for retrieving examples")
        dfsl_subgroup.add_argument("--examples_dataset_path", type=str, help="Examples dataset path for retrieving examples")
        
        return parser
    
    @staticmethod
    def create_from_args(args) -> "PythiaAgent":
        kg: KnowledgeGraphs = None
        for x in KnowledgeGraphs:
            if x.name.__class__.lower() == args.graph.lower():
                kg = x
                break
        if kg is None:
            raise ValueError(f"Unknown knowledge graph: {args.graph}. Available options are: {[x.name for x in KnowledgeGraphs]}")
        kg.load()
        
        agent = PythiaAgent(kg, 
                            db_dataset_name=args.examples_dataset,
                            db_dataset_path=args.examples_dataset_path)
        
        return agent


if __name__ == "__main__":    
    import argparse
    from src.logging import create_console_logger
    
    parser = PythiaArgParser()
    parser = PythiaAgent.fill_parse_args(parser, graph_required=False)
    
    args = parser.parse_args()
    
    create_console_logger()
    
    kg=KnowledgeGraphs.DBPEDIA10
    kg.load()
    
    agent = PythiaAgent(kg)
    
    results = agent.answer("Which is the highest mountain in Germany?", stopwatch=False, sparql_only=False)
    for key, value in results.items():
        if key != "messages":
            print(f"{key}: {value}")
            
    results = agent.answer("Which airline has the most popular frequent flyer program?", stopwatch=False, sparql_only=False)
    for key, value in results.items():
        if key != "messages":
            print(f"{key}: {value}")