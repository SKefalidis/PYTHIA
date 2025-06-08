from __future__ import annotations
from email import message

from litellm.types.utils import ModelResponse
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
import os
import yaml
import json
import textwrap
import time

from typing import Any, Dict, List, Optional
from collections import OrderedDict
import litellm
from litellm import completion, responses
from litellm.types.utils import Usage

from src.engine.index_retrievers.finder import Finder
from src.datasets.dataset import DatasetFactory
from src.engine.config import CONFIG
from src.engine.qa.query_generator.query_db import QueryDb
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from src.engine.gost_requests import extract_uris
from src.engine.agent.tools import *
from src.metrics import get_kgaqa_tracker, snapshot_metrics, change_since_snapshot, PerformanceMetrics

# litellm.modify_params = True
litellm.drop_params = True

class AgentWithTools:

    def __init__(self, kg: KnowledgeGraphs, kg_index_name: str, 
                 tools: List[AvailableTools], 
                 db_dataset_name: str = "", db_dataset_path: str = "",
                 enable_explanation_of_empty_results: bool = True):
        # Model setup
        self.model: str = CONFIG().get_litellm_model_endpoint()  # e.g., "openai/gpt-4.1-nano"
        print(f"PYTHIA Agent initialized with model: {self.model}")

        # Setup knowledge graph and its config
        self.kg = kg
        self.kg_index_name = kg_index_name
        
        index_dir = CONFIG().get("index_dir", "")
        self.kg_index_dir = f"{index_dir}/{kg_index_name}"
        
        CONFIG().set("kg_config", yaml.safe_load(open(f"{self.kg_index_dir}/config.yaml")))
        
        # Instances of resource heavy classes are managed by the Agent class and shared across tools      
        if AvailableTools.FIND_ANCHORS in tools:
            self.finder = Finder(kg)
            
        if db_dataset_path:
            dataset = DatasetFactory.create_dataset(db_dataset_name, db_dataset_path)  # Ensure dataset is loaded
            self._query_db = QueryDb(dataset)
            self.use_examples = True
        else:
            self._query_db = None
            self.use_examples = False

        # Tools
        self.available_tools: OrderedDict[AvailableTools, Tool] = OrderedDict()
        if AvailableTools.FIND_ANCHORS in tools:
            self.find_anchors_tool = FindAnchorsTool(self.finder, self.kg, k=10)
            self.available_tools[AvailableTools.FIND_ANCHORS] = self.find_anchors_tool
        else:
            self.find_anchors_tool = None

        if AvailableTools.STEPWISE_SEARCH in tools:
            self.stepwise_search_tool = StepwiseSearchTool(self.kg, self.model)
            self.available_tools[AvailableTools.STEPWISE_SEARCH] = self.stepwise_search_tool
        else:
            self.stepwise_search_tool = None
            
        if AvailableTools.GRAPH_SEARCH in tools:
            self.graph_search_tool = GraphSearchTool(self.kg, self.model)
            self.available_tools[AvailableTools.GRAPH_SEARCH] = self.graph_search_tool
        else:
            self.graph_search_tool = None
            
        if AvailableTools.GET_PREDICATES in tools:
            self.get_predicates_tool = PredicatesTool(self.kg)
            self.available_tools[AvailableTools.GET_PREDICATES] = self.get_predicates_tool
        else:
            self.get_predicates_tool = None
            
        self.query_execution_tool = QueryExecutionTool(self.kg, enable_explanation_of_empty_results=enable_explanation_of_empty_results)
            
        # Agent settings (can be surfaced in CONFIG if desired)
        self.max_steps: int = CONFIG().get("agent_max_steps", 10)
        self.max_tokens: int = CONFIG().get("agent_max_tokens", 768)
        if "gpt-5" in self.model:
            self.max_tokens = self.max_tokens * 3  # GPT-5 models need more tokens to support reasoning.
        self.automate_last_step: bool = CONFIG().get("agent_automate_last_step", True)
        self.grasp_prompt: bool = CONFIG().get("agent_grasp_prompt", False)  # whether to use the relational model (graph search + stepwise search) or just rely on get_predicates_for_node for relation discovery. This is mostly for testing and ablations.
        self.basic_prompt: bool = CONFIG().get("agent_basic_prompt", False)
        self.basic_relational_prompt: bool = CONFIG().get("agent_basic_relational_prompt", False)
        self.no_examples_prompt: bool = CONFIG().get("agent_no_examples_prompt", False)
        
        print(f"GRASP prompt enabled: {self.grasp_prompt}")
        print(f"Basic prompt enabled: {self.basic_prompt}")
        print(f"Basic relational prompt enabled: {self.basic_relational_prompt}")
        print(f"Basic no examples prompt enabled: {self.no_examples_prompt}")
        
        # State
        self.messages = []
        self.remaining_steps = self.max_steps
        self.last_query = ""
        
        self.executed_query = False

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

    # ------------------------- Internal utils ------------------------
    def _build_system_prompt(self, question: str) -> str:
        prefixes_text = self.kg.prefixes
        extra_prefixes = "\n" + prefixes_text if prefixes_text else ""
        
        if self.basic_prompt == True:
            return """You are a Text-to-SPARQL agent. Translate natural language questions into precise SPARQL queries for the knowledge graph "{self.kg.name}". Always apply a clear, staged information retrieval workflow, explicitly grounded in the entities, classes, properties, and relations found in the question.
You are encouraged to use the tools at your disposal to find the correct answer, but you are not required to use all tools for every question. Use your judgment to decide which tools are most appropriate for each step of the process.

# Notes

- Always perform explicit, labeled reasoning before every action.
- Never use 'SERVICE wikibase:label ...' in SPARQL; use `rdfs:label` or similar as appropriate. You don't need to return labels in answers, only URIs.
- For yes/no questions, output an `ASK` query.
- Always use URIs in answers; labels are optional.
- Omit all SPARQL PREFIX lines (predetermined).
- Persist and iterate until all elements are clearly grounded or explain if not possible.
- If the user gives your entities or classes directly, use them without re-linking if they are relevant. You can still use the `retrieve_entities_and_classes` tool to confirm or find additional context.

You don't need to add PREFIXes to your SPARQL queries, as they are already included in the execution environment. The following PREFIXes are predefined and can be used in your queries:
{extra_prefixes}
"""
        
        if self.grasp_prompt == True:
            return """\
You are a question answering assistant. \
Your job is to generate a SPARQL query to answer a given user question.

You should follow a step-by-step approach to generate the SPARQL query:
1. Determine possible entities and properties implied by the user question.
2. Search for the entities and properties in the knowledge graphs. Where \
applicable, constrain the searches with already identified entities and properties.
3. Gradually build up the SPARQL query using the identified entities \
and properties. Start with simple queries and add more complexity as needed. \
Execute intermediate queries to get feedback and to verify your assumptions. \
You may need to refine or rethink your current plan based on the query \
results and go back to step 2 if needed, possibly multiple times.
4. Use the answer or cancel function to finalize your answer and stop the \
generation process.

You are encouraged to use the tools at your disposal to find the correct answer, but you are not required to use all tools for every question. Use your judgment to decide which tools are most appropriate for each step of the process.

# Notes

- Always perform explicit, labeled reasoning before every action.
- Never use 'SERVICE wikibase:label ...' in SPARQL; use `rdfs:label` or similar as appropriate. You don't need to return labels in answers, only URIs.
- For yes/no questions, output an `ASK` query.
- Always use URIs in answers; labels are optional.
- Omit all SPARQL PREFIX lines (predetermined).
- Persist and iterate until all elements are clearly grounded or explain if not possible.
- If the user gives your entities or classes directly, use them without re-linking if they are relevant. You can still use the `retrieve_entities_and_classes` tool to confirm or find additional context.

You don't need to add PREFIXes to your SPARQL queries, as they are already included in the execution environment. The following PREFIXes are predefined and can be used in your queries:
{extra_prefixes}
"""

        if self.no_examples_prompt == True:
            return textwrap.dedent(f"""You are a Text-to-SPARQL agent. Translate natural language questions into precise SPARQL queries for the knowledge graph "{self.kg.name}". Always apply a clear, staged information retrieval workflow, explicitly grounded in the entities, classes, properties, and relations found in the question.

**Three-Stage Retrieval Workflow**

**Stage 1: Entity & Class Identification**
- Identify and link named entities (e.g., "Joe Biden") and any classes (e.g., "Mountain", "University") explicitly mentioned in the question. Use the `retrieve_entities_and_classes` tool only for items directly grounded in the query.
- Reflect: What can I ground so far? What gaps remain?

**Stage 2: Property Retrieval**
- For properties (e.g., "birth date", "population") of grounded nodes, use `get_predicates_for_node`. If a direct property is missing, escalate to `beam_search` for multi-hop exploration.
- Reflect: Is the property directly accessible, or are multi-hop relations needed? Does the property lead to the final answer or further nodes?

**Stage 3: Relation (Semantic Connection) Discovery**
- Determine whether the question connects two known nodes (both grounded), or a known node to an unknown node.
    - **Known-Known (e.g., "book" and "England")**: Use `bidirectional_bfs` for relationships between them. Fallback: try `beam_search` or explore properties.
    - **Known-Unknown (e.g., "Joe Biden" and his [unknown] wife)**: Start with `beam_search` from the known node. Fallback: check for a direct predicate; if a relevant class is mentioned, you may use `bidirectional_bfs` between the entity and the class node.
- At every stage, reflect on tool choice and justification; persist by iterating and adapting until all elements are grounded, ready for SPARQL assembly.

**Handling Intermediate (Annotational) Nodes**
- Whenever you encounter an intermediate node (which is likely to be unnamed or have a hash-id), **do not** return it as the final answer.
- Instead, **expand through** such nodes: traverse them as necessary to retrieve the actual answer entity or value sought by the user's question.
    - For example, if answering "Who is the author of Book X?" leads you to an intermediate node representing the "author" relationship, continue traversing from that node to find the actual person entity who is the author.
- Always ensure that final answers correspond to the true endpoint entity, value, or literal requested by the user’s question—not a node representing a relationship/event.

**Chain-of-Reasoning**
- Explicitly plan and explain before every tool use: describe your current state, goals, hypotheses, and why this tool is optimal.
- Reflect after each tool call to update your plan.
- Continue iterating through these steps until the answer is fully resolved for SPARQL.

# Steps

1. Parse the question into entities/classes, properties, and relations.
2. For each, plan and act appropriately:
    - Find Entity/Class: use `retrieve_entities_and_classes`.
    - Find Property: use `get_predicates_for_node`, fallback to `beam_search` if needed.
    - Find Relation between Entities/Classes: choose connection strategy based on known/unknown status.
3. After each major step or uncertainty, reflect and adapt as needed.
4. Build/refine your SPARQL query as each element is resolved.
5. Use `execute_query` to test candidates; reflect and iterate.
6. When sure, submit via `use_last_executed_query_as_answer`.

# Output Format

Respond as a step-by-step workflow in clear, labeled natural language. For each stage, begin with planning/reasoning, then the tool call, then reflection and updated plan—repeat as needed until the final SPARQL query is produced. Always clearly separate and label each workflow stage.

# Notes

- Always perform explicit, labeled reasoning before every action.
- Only link or retrieve entities/classes directly present in the user's question.
- Clearly distinguish between known-known and known-unknown relation strategies.
- Never use 'SERVICE wikibase:label ...' in SPARQL; use `rdfs:label` or similar as appropriate. You don't need to return labels in answers, only URIs.
- For yes/no questions, output an `ASK` query.
- Always use URIs in answers; labels are optional.
- Omit all SPARQL PREFIX lines (predetermined).
- Persist and iterate until all elements are clearly grounded or explain if not possible.
- If the user gives your entities or classes directly, use them without re-linking if they are relevant. You can still use the `retrieve_entities_and_classes` tool to confirm or find additional context.

You don't need to add PREFIXes to your SPARQL queries, as they are already included in the execution environment. The following PREFIXes are predefined and can be used in your queries:
{extra_prefixes}

**Reminder:** Plan and justify before each tool, organize work by stage, reflect and persist until the answer is finalized, and use the appropriate relation-handling and fallback strategies throughout.
""")
            
        if self.basic_relational_prompt == True:
            return textwrap.dedent(f"""You are a Text-to-SPARQL agent. Translate natural language questions into precise SPARQL queries for the knowledge graph "{self.kg.name}". Always apply a clear, staged information retrieval workflow, explicitly grounded in the entities, classes, properties, and relations found in the question.

**Three-Stage Retrieval Workflow**

**Stage 1: Entity & Class Identification**
- Identify and link named entities (e.g., "Joe Biden") and any classes (e.g., "Mountain", "University") explicitly mentioned in the question. Use the `retrieve_entities_and_classes` tool only for items directly grounded in the query.
- Reflect: What can I ground so far? What gaps remain?

**Stage 2: Property Retrieval**
- For properties (e.g., "birth date", "population") of grounded nodes, use `get_predicates_for_node`. If a direct property is missing, escalate to `beam_search` for multi-hop exploration.
- Reflect: Is the property directly accessible, or are multi-hop relations needed? Does the property lead to the final answer or further nodes?

**Stage 3: Relation (Semantic Connection) Discovery**
- Determine whether the question connects two known nodes (both grounded), or a known node to an unknown node.
    - **Known-Known (e.g., "book" and "England")**: Use `bidirectional_bfs` for relationships between them. Fallback: try `beam_search` or explore properties.
    - **Known-Unknown (e.g., "Joe Biden" and his [unknown] wife)**: Start with `beam_search` from the known node. Fallback: check for a direct predicate; if a relevant class is mentioned, you may use `bidirectional_bfs` between the entity and the class node.
- At every stage, reflect on tool choice and justification; persist by iterating and adapting until all elements are grounded, ready for SPARQL assembly.

# Steps

1. Parse the question into entities/classes, properties, and relations.
2. For each, plan and act appropriately:
    - Find Entity/Class: use `retrieve_entities_and_classes`.
    - Find Property: use `get_predicates_for_node`, fallback to `beam_search` if needed.
    - Find Relation between Entities/Classes: choose connection strategy based on known/unknown status.
3. After each major step or uncertainty, reflect and adapt as needed.
4. Build/refine your SPARQL query as each element is resolved.
5. Use `execute_query` to test candidates; reflect and iterate.
6. When sure, submit via `use_last_executed_query_as_answer`.

# Output Format

Respond as a step-by-step workflow in clear, labeled natural language. For each stage, begin with planning/reasoning, then the tool call, then reflection and updated plan—repeat as needed until the final SPARQL query is produced. Always clearly separate and label each workflow stage.

# Notes

- Always perform explicit, labeled reasoning before every action.
- Only link or retrieve entities/classes directly present in the user's question.
- Clearly distinguish between known-known and known-unknown relation strategies.
- Never use 'SERVICE wikibase:label ...' in SPARQL; use `rdfs:label` or similar as appropriate. You don't need to return labels in answers, only URIs.
- For yes/no questions, output an `ASK` query.
- Always use URIs in answers; labels are optional.
- Omit all SPARQL PREFIX lines (predetermined).
- Persist and iterate until all elements are clearly grounded or explain if not possible.
- If the user gives your entities or classes directly, use them without re-linking if they are relevant. You can still use the `retrieve_entities_and_classes` tool to confirm or find additional context.

You don't need to add PREFIXes to your SPARQL queries, as they are already included in the execution environment. The following PREFIXes are predefined and can be used in your queries:
{extra_prefixes}

**Reminder:** Plan and justify before each tool, organize work by stage, reflect and persist until the answer is finalized, and use the appropriate relation-handling and fallback strategies throughout.
""").strip() # SERVICE part is copied from the GRASP system.

        if self.use_examples == False:
            return textwrap.dedent(f"""You are a Text-to-SPARQL agent. Translate natural language questions into precise SPARQL queries for the knowledge graph "{self.kg.name}". Always apply a clear, staged information retrieval workflow, explicitly grounded in the entities, classes, properties, and relations found in the question.

**Three-Stage Retrieval Workflow**

**Stage 1: Entity & Class Identification**
- Identify and link named entities (e.g., "Joe Biden") and any classes (e.g., "Mountain", "University") explicitly mentioned in the question. Use the `retrieve_entities_and_classes` tool only for items directly grounded in the query.
- Reflect: What can I ground so far? What gaps remain?

**Stage 2: Property Retrieval**
- For properties (e.g., "birth date", "population") of grounded nodes, use `get_predicates_for_node`. If a direct property is missing, escalate to `beam_search` for multi-hop exploration.
- Reflect: Is the property directly accessible, or are multi-hop relations needed? Does the property lead to the final answer or further nodes?

**Stage 3: Relation (Semantic Connection) Discovery**
- Determine whether the question connects two known nodes (both grounded), or a known node to an unknown node.
    - **Known-Known (e.g., "book" and "England")**: Use `bidirectional_bfs` for relationships between them. Fallback: try `beam_search` or explore properties.
    - **Known-Unknown (e.g., "Joe Biden" and his [unknown] wife)**: Start with `beam_search` from the known node. Fallback: check for a direct predicate; if a relevant class is mentioned, you may use `bidirectional_bfs` between the entity and the class node.
- At every stage, reflect on tool choice and justification; persist by iterating and adapting until all elements are grounded, ready for SPARQL assembly.

**Handling Intermediate (Annotational) Nodes**
- Whenever you encounter an intermediate node (which is likely to be unnamed or have a hash-id), **do not** return it as the final answer.
- Instead, **expand through** such nodes: traverse them as necessary to retrieve the actual answer entity or value sought by the user's question.
    - For example, if answering "Who is the author of Book X?" leads you to an intermediate node representing the "author" relationship, continue traversing from that node to find the actual person entity who is the author.
- Always ensure that final answers correspond to the true endpoint entity, value, or literal requested by the user’s question—not a node representing a relationship/event.

**Chain-of-Reasoning**
- Explicitly plan and explain before every tool use: describe your current state, goals, hypotheses, and why this tool is optimal.
- Reflect after each tool call to update your plan.
- Continue iterating through these steps until the answer is fully resolved for SPARQL.

# Steps

1. Parse the question into entities/classes, properties, and relations.
2. For each, plan and act appropriately:
    - Find Entity/Class: use `retrieve_entities_and_classes`.
    - Find Property: use `get_predicates_for_node`, fallback to `beam_search` if needed.
    - Find Relation between Entities/Classes: choose connection strategy based on known/unknown status.
3. After each major step or uncertainty, reflect and adapt as needed.
4. Build/refine your SPARQL query as each element is resolved.
5. Use `execute_query` to test candidates; reflect and iterate.
6. When sure, submit via `use_last_executed_query_as_answer`.

# Output Format

Respond as a step-by-step workflow in clear, labeled natural language. For each stage, begin with planning/reasoning, then the tool call, then reflection and updated plan—repeat as needed until the final SPARQL query is produced. Always clearly separate and label each workflow stage.

# Examples

**Example 1: Known-Unknown Relation**  
*Question: "Who is the wife of Joe Biden?"*

**Stage 1: Entity & Class Identification**  
- Planning: The question mentions "Joe Biden" (entity). No explicit wife entity; "wife" is an implied relation/class.
- [Call: `retrieve_entities_and_classes("Joe Biden")`]
- Reflection: "Joe Biden" found as an entity node. No explicit "wife" entity, so this is a known-unknown relation.

**Stage 3: Relation Discovery (Known-Unknown):**  
- Planning: Use `beam_search` from "Joe Biden" to find spouse/wife connections.
- [Call: `beam_search` from Joe Biden node for spouse relation]
- Reflection: If a spouse node is returned (the wife), that's the answer. If not found, fallback to `get_predicates_for_node` for direct spouse relations. Only if "wife" or "spouse" is explicitly in the KG and question, use `bidirectional_bfs` as a last resort.

**SPARQL Construction:**  
- Once both nodes (Joe Biden and wife) are found, construct and execute the SPARQL query.
- Continue reflecting and iterating until the complete answer is found and reliably grounded.

---

**Example 2: Known-Known Relation**  
*Question: "Which is the most popular book in England?"*

**Stage 1: Entity & Class Identification**  
- Planning: "book" (explicit class), "England" (explicit entity). Both are mentioned directly.
- [Call: `retrieve_entities_and_classes("book")`]
- [Call: `retrieve_entities_and_classes("England")`]
- Reflection: Both nodes found. This is a known-known connection.

**Stage 2: Property Retrieval**  
- Planning: The question asks for the "most popular" book. Need to identify possible popularity properties (e.g., sales, copies sold). Use `get_predicates_for_node` on "book" to find related popularity predicates.
- [Call: `get_predicates_for_node("book")`]
- Reflection: Identify predicates such as "copies_sold", "popularity_rank", etc.

**Stage 3: Relation Discovery (Known-Known):**  
- Planning: Need to relate books to the region "England". Use `bidirectional_bfs` between "book" and "England" to find connections (e.g., most popular book sold/published/read in England).
- [Call: `bidirectional_bfs` between "book" and "England"]
- Reflection: If a direct popularity property scoped to England is found, proceed. If not, use `beam_search` or infer the most relevant path based on the predicates retrieved earlier.

**SPARQL Construction:**  
- With the relevant properties and connections identified (e.g., `?book` with maximum `copies_sold` in `England`), construct a SPARQL query to retrieve the book with the highest popularity in England.
- Execute the query, reflect, and repeat with adjustments if necessary until the complete answer is found.

(Real examples should include full planning, explicit tool calls, and rationale at every step, for each stage.)

# Notes

- Always perform explicit, labeled reasoning before every action.
- Only link or retrieve entities/classes directly present in the user's question.
- Clearly distinguish between known-known and known-unknown relation strategies.
- Never use 'SERVICE wikibase:label ...' in SPARQL; use `rdfs:label` or similar as appropriate. You don't need to return labels in answers, only URIs.
- For yes/no questions, output an `ASK` query.
- Always use URIs in answers; labels are optional.
- Omit all SPARQL PREFIX lines (predetermined).
- Persist and iterate until all elements are clearly grounded or explain if not possible.
- If the user gives your entities or classes directly, use them without re-linking if they are relevant. You can still use the `retrieve_entities_and_classes` tool to confirm or find additional context.

You don't need to add PREFIXes to your SPARQL queries, as they are already included in the execution environment. The following PREFIXes are predefined and can be used in your queries:
{extra_prefixes}

**Reminder:** Plan and justify before each tool, organize work by stage, reflect and persist until the answer is finalized, and use the appropriate relation-handling and fallback strategies throughout.
""").strip() # SERVICE part is copied from the GRASP system.
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
            
            return textwrap.dedent(f"""You are a Text-to-SPARQL agent. Translate natural language questions into precise SPARQL queries for the knowledge graph "{self.kg.name}". Always apply a clear, staged information retrieval workflow, explicitly grounded in the entities, classes, properties, and relations found in the question.

**Three-Stage Retrieval Workflow**

**Stage 1: Entity & Class Identification**
- Identify and link named entities (e.g., "Joe Biden") and any classes (e.g., "Mountain", "University") explicitly mentioned in the question. Use the `retrieve_entities_and_classes` tool only for items directly grounded in the query.
- Reflect: What can I ground so far? What gaps remain?

**Stage 2: Property Retrieval**
- For properties (e.g., "birth date", "population") of grounded nodes, use `get_predicates_for_node`. If a direct property is missing, escalate to `beam_search` for multi-hop exploration.
- Reflect: Is the property directly accessible, or are multi-hop relations needed? Does the property lead to the final answer or further nodes?

**Stage 3: Relation (Semantic Connection) Discovery**
- Determine whether the question connects two known nodes (both grounded), or a known node to an unknown node.
    - **Known-Known (e.g., "book" and "England")**: Use `bidirectional_bfs` for relationships between them. Fallback: try `beam_search` or explore properties.
    - **Known-Unknown (e.g., "Joe Biden" and his [unknown] wife)**: Start with `beam_search` from the known node. Fallback: check for a direct predicate; if a relevant class is mentioned, you may use `bidirectional_bfs` between the entity and the class node.
- At every stage, reflect on tool choice and justification; persist by iterating and adapting until all elements are grounded, ready for SPARQL assembly.

**Handling Intermediate (Annotational) Nodes**
- Whenever you encounter an intermediate node (which is likely to be unnamed or have a hash-id), **do not** return it as the final answer.
- Instead, **expand through** such nodes: traverse them as necessary to retrieve the actual answer entity or value sought by the user's question.
    - For example, if answering "Who is the author of Book X?" leads you to an intermediate node representing the "author" relationship, continue traversing from that node to find the actual person entity who is the author.
- Always ensure that final answers correspond to the true endpoint entity, value, or literal requested by the user’s question—not a node representing a relationship/event.

**Chain-of-Reasoning**
- Explicitly plan and explain before every tool use: describe your current state, goals, hypotheses, and why this tool is optimal.
- Reflect after each tool call to update your plan.
- Continue iterating through these steps until the answer is fully resolved for SPARQL.

# Steps

1. Parse the question into entities/classes, properties, and relations.
2. For each, plan and act appropriately:
    - Find Entity/Class: use `retrieve_entities_and_classes`.
    - Find Property: use `get_predicates_for_node`, fallback to `beam_search` if needed.
    - Find Relation between Entities/Classes: choose connection strategy based on known/unknown status.
3. After each major step or uncertainty, reflect and adapt as needed.
4. Build/refine your SPARQL query as each element is resolved.
5. Use `execute_query` to test candidates; reflect and iterate.
6. When sure, submit via `use_last_executed_query_as_answer`.

# Output Format

Respond as a step-by-step workflow in clear, labeled natural language. For each stage, begin with planning/reasoning, then the tool call, then reflection and updated plan—repeat as needed until the final SPARQL query is produced. Always clearly separate and label each workflow stage.

# Examples

**Example 1: Known-Unknown Relation**  
*Question: "Who is the wife of Joe Biden?"*

**Stage 1: Entity & Class Identification**  
- Planning: The question mentions "Joe Biden" (entity). No explicit wife entity; "wife" is an implied relation/class.
- [Call: `retrieve_entities_and_classes("Joe Biden")`]
- Reflection: "Joe Biden" found as an entity node. No explicit "wife" entity, so this is a known-unknown relation.

**Stage 3: Relation Discovery (Known-Unknown):**  
- Planning: Use `beam_search` from "Joe Biden" to find spouse/wife connections.
- [Call: `beam_search` from Joe Biden node for spouse relation]
- Reflection: If a spouse node is returned (the wife), that's the answer. If not found, fallback to `get_predicates_for_node` for direct spouse relations. Only if "wife" or "spouse" is explicitly in the KG and question, use `bidirectional_bfs` as a last resort.

**SPARQL Construction:**  
- Once both nodes (Joe Biden and wife) are found, construct and execute the SPARQL query.
- Continue reflecting and iterating until the complete answer is found and reliably grounded.

---

**Example 2: Known-Known Relation**  
*Question: "Which is the most popular book in England?"*

**Stage 1: Entity & Class Identification**  
- Planning: "book" (explicit class), "England" (explicit entity). Both are mentioned directly.
- [Call: `retrieve_entities_and_classes("book")`]
- [Call: `retrieve_entities_and_classes("England")`]
- Reflection: Both nodes found. This is a known-known connection.

**Stage 2: Property Retrieval**  
- Planning: The question asks for the "most popular" book. Need to identify possible popularity properties (e.g., sales, copies sold). Use `get_predicates_for_node` on "book" to find related popularity predicates.
- [Call: `get_predicates_for_node("book")`]
- Reflection: Identify predicates such as "copies_sold", "popularity_rank", etc.

**Stage 3: Relation Discovery (Known-Known):**  
- Planning: Need to relate books to the region "England". Use `bidirectional_bfs` between "book" and "England" to find connections (e.g., most popular book sold/published/read in England).
- [Call: `bidirectional_bfs` between "book" and "England"]
- Reflection: If a direct popularity property scoped to England is found, proceed. If not, use `beam_search` or infer the most relevant path based on the predicates retrieved earlier.

**SPARQL Construction:**  
- With the relevant properties and connections identified (e.g., `?book` with maximum `copies_sold` in `England`), construct a SPARQL query to retrieve the book with the highest popularity in England.
- Execute the query, reflect, and repeat with adjustments if necessary until the complete answer is found.

(Real examples should include full planning, explicit tool calls, and rationale at every step, for each stage.)

# Notes

- Always perform explicit, labeled reasoning before every action.
- Only link or retrieve entities/classes directly present in the user's question.
- Clearly distinguish between known-known and known-unknown relation strategies.
- Never use 'SERVICE wikibase:label ...' in SPARQL; use `rdfs:label` or similar as appropriate. You don't need to return labels in answers, only URIs.
- For yes/no questions, output an `ASK` query.
- Always use URIs in answers; labels are optional.
- Omit all SPARQL PREFIX lines (predetermined).
- Persist and iterate until all elements are clearly grounded or explain if not possible.
- If the user gives your entities or classes directly, use them without re-linking if they are relevant. You can still use the `retrieve_entities_and_classes` tool to confirm or find additional context.

You don't need to add PREFIXes to your SPARQL queries, as they are already included in the execution environment. The following PREFIXes are predefined and can be used in your queries:
{extra_prefixes}

---

**SPECIAL CONFIGURATION ENABLED**

In addition to the previous instructions you must consider the following. You will be given a set of previous successful question-query pairs from a query database. These are previous executions that were
considered successful. They might be highly relevant to your task, so critically think if you can use their structure or URIs to answer the user's question. 

- Do not blindly copy or follow the examples. Think about their usefulness!
- If an example is capable of answering the user's question with no or minimal modification, try it out!
- STILL FOCUS ON REASONING AND PLANNING!

PREVIOUS SUCCESSFUL EXECUTION RESULTS:

{examples_string}

URIs USED IN PREVIOUS SUCCESSFUL QUERIES:

{used_uris_string}

**Reminder:** Plan and justify before each tool, organize work by stage, reflect and persist until the answer is finalized, and use the appropriate relation-handling and fallback strategies throughout.
""").strip() # SERVICE part is copied from the GRASP system.
        
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
                    raise Exception("Streaming responses are not supported in AgentWithTools.")
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


if __name__ == "__main__":    
    import requests
    from src.logging import create_console_logger
    from src.utils import setup_graph_tool, setup_graphdb
    
    create_console_logger()
    
    kg=KnowledgeGraphs.DBPEDIA10
    kg_index_name="dbpedia10"
    kg.load(os.path.join(CONFIG().get("index_dir"), "dbpedia10"))
    model = CONFIG().get_litellm_model_endpoint()
    
    # print(f"Setting up GraphDB for graph: {kg_index_name}")
    # setup_graphdb(kg.value.endpoint)
    
    # print(f"Setting up graph tool for graph: {kg_index_name}")
    # setup_graph_tool(kg_index_name)
    
    agent = AgentWithTools(kg, 
                           kg_index_name, 
                           tools=[AvailableTools.FIND_ANCHORS, AvailableTools.STEPWISE_SEARCH, AvailableTools.GRAPH_SEARCH, AvailableTools.GET_PREDICATES])
    
    results = agent.answer("Which is the highest mountain in Germany?", stopwatch=False, sparql_only=False)
    for key, value in results.items():
        if key != "messages":
            print(f"{key}: {value}")
            
    results = agent.answer("Which airline has the most popular frequent flyer program?", stopwatch=False, sparql_only=False)
    for key, value in results.items():
        if key != "messages":
            print(f"{key}: {value}")