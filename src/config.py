import yaml
import re
import os
import argparse
from typing_extensions import override
from pathlib import Path
from pydantic import BaseModel

# ---------------------------------
# ----- CONFIGURATION CLASSES -----
# ---------------------------------

class GeneralSettings(BaseModel):
    version: str
    index_dir: str

class AgentGeneralSettings(BaseModel):
    max_tokens: int
    max_steps: int
    automate_last_step: bool
    
    def fill_parse_args(self, parser: argparse.ArgumentParser) -> None:
        """Fill the argparse parser with the agent general settings."""
        agent_subgroup = parser.add_argument_group(title="PYTHIA - General Agent Settings", description="General settings for the main agent.")
        agent_subgroup.add_argument("--max_tokens", type=int, default=self.max_tokens, help="Maximum number of tokens for the agent.")
        agent_subgroup.add_argument("--max_steps", type=int, default=self.max_steps, help="Maximum number of steps for the agent.")
        agent_subgroup.add_argument("--automate_last_step", help="Whether to automate the last step for the agent.")
        
    def set_from_args(self, args: argparse.Namespace) -> None:
        """Set the agent general settings from the parsed command line arguments."""
        self.max_tokens = getattr(args, "max_tokens", self.max_tokens)
        self.max_steps = getattr(args, "max_steps", self.max_steps)
        self.automate_last_step = getattr(args, "automate_last_step", self.automate_last_step)
    
class AgentPromptSettings(BaseModel):
    basic_prompt: bool
    basic_relational_prompt: bool
    no_examples_prompt: bool
    grasp_prompt: bool
    
    def fill_parse_args(self, parser: argparse.ArgumentParser) -> None:
        """Fill the argparse parser with the agent prompt settings."""
        prompt_subgroup = parser.add_argument_group(title="PYTHIA - Prompt Settings", description="Settings for the main agent's prompt.")
        prompt_subgroup.add_argument("--basic_prompt", action='store_true', help="Use basic prompt for the agent.")
        prompt_subgroup.add_argument("--basic_relational_prompt", action='store_true', help="Use basic relational prompt for the agent.")
        prompt_subgroup.add_argument("--no_examples_prompt", action='store_true', help="Use no examples prompt for the agent.")
        prompt_subgroup.add_argument("--grasp_prompt", action='store_true', help="Use GRASP prompt for the agent.")
        
    def set_from_args(self, args: argparse.Namespace) -> None:
        """Set the agent prompt settings from the parsed command line arguments."""
        self.basic_prompt = getattr(args, "basic_prompt", self.basic_prompt)
        self.basic_relational_prompt = getattr(args, "basic_relational_prompt", self.basic_relational_prompt)
        self.no_examples_prompt = getattr(args, "no_examples_prompt", self.no_examples_prompt)
        self.grasp_prompt = getattr(args, "grasp_prompt", self.grasp_prompt)
    
class AgentToolsSettings(BaseModel):
    no_find: bool
    no_graph_search: bool
    no_stepwise_search: bool
    no_get_predicates: bool
    no_empty_query_investigation: bool
    
    def fill_parse_args(self, parser: argparse.ArgumentParser) -> None:
        """Fill the argparse parser with the agent tools settings."""
        tools_subgroup = parser.add_argument_group(title="PYTHIA - Tools Settings", description="Settings for the main agent's tools.")
        tools_subgroup.add_argument("--no_find", action='store_true', help="Disable the 'find' tool for the agent.")
        tools_subgroup.add_argument("--no_graph_search", action='store_true', help="Disable the 'graph search' tool for the agent.")
        tools_subgroup.add_argument("--no_stepwise_search", action='store_true', help="Disable the 'stepwise search' tool for the agent.")
        tools_subgroup.add_argument("--no_get_predicates", action='store_true', help="Disable the 'get predicates' tool for the agent.")
        tools_subgroup.add_argument("--no_empty_query_investigation", action='store_true', help="Disable the 'empty query investigation' tool for the agent.")
        
    def set_from_args(self, args: argparse.Namespace) -> None:
        """Set the agent tools settings from the parsed command line arguments."""
        self.no_find = getattr(args, "no_find", self.no_find)
        self.no_graph_search = getattr(args, "no_graph_search", self.no_graph_search)
        self.no_stepwise_search = getattr(args, "no_stepwise_search", self.no_stepwise_search)
        self.no_empty_query_investigation = getattr(args, "no_empty_query_investigation", self.no_empty_query_investigation)
    
class AgentSettings(BaseModel):                                                                                                                                                                          
    settings: AgentGeneralSettings                                                                                                                                                                       
    prompt: AgentPromptSettings                                                                                                                                                                          
    tools: AgentToolsSettings
    
    def fill_parse_args(self, parser: argparse.ArgumentParser) -> None:
        """Fill the argparse parser with the agent settings."""
        self.settings.fill_parse_args(parser)
        self.prompt.fill_parse_args(parser)
        self.tools.fill_parse_args(parser)
        
    def set_from_args(self, args: argparse.Namespace) -> None:
        """Set the agent settings from the parsed command line arguments."""
        self.settings.set_from_args(args)
        self.prompt.set_from_args(args)
        self.tools.set_from_args(args)
    
class QueryDatabaseSettings(BaseModel):
    top_k: int
    
    def fill_parse_args(self, parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
        """Fill the argparse parser with the query database settings."""
        query_db_subgroup = parser.add_argument_group(title="Query Database Settings", description="Settings for the query database.")
        query_db_subgroup.add_argument("--top_k", type=int, default=self.top_k, help="Number of top results to retrieve from the query database.")
        return query_db_subgroup
        
    def set_from_args(self, args: argparse.Namespace) -> None:
        """Set the query database settings from the parsed command line arguments."""
        self.top_k = getattr(args, "top_k", self.top_k)

class LlmSettings(BaseModel):
    provider: str
    model: str
    
    def get_litellm_model_endpoint(self) -> str:
        """Returns the Litellm endpoint URL based on the configuration."""
        return f"{self.provider}/{self.model}"
    
    def fill_parse_args(self, parser: argparse.ArgumentParser) -> None:
        """Fill the argparse parser with the LLM settings."""
        llm_subgroup = parser.add_argument_group(title="LLM Settings", description="Settings for the LLM.")
        llm_subgroup.add_argument("--provider", type=str, default=self.provider, help="Provider for the LLM.")
        llm_subgroup.add_argument("--model", type=str, default=self.model, help="Model for the LLM.")
        
    def set_from_args(self, args: argparse.Namespace) -> None:
        """Set the LLM settings from the parsed command line arguments."""
        self.provider = getattr(args, "provider", self.provider)
        self.model = getattr(args, "model", self.model)
        
class GraphSearchSettings(BaseModel):
    endpoint: str
    filter_classes: bool
    timeout: int|None = None
    load_on_construction: bool           # whether to load the graph search index on construction of the GraphSearchManager.
    crash_on_misconfigured_server: bool  # whether to crash the program if the graph search server is misconfigured (i.e., does not have the correct graph search index loaded)
    
class AppGeneralSettings(BaseModel):                                                                                                                                                                     
    execute_sparql: bool
    
    def fill_parse_args(self, parser: argparse.ArgumentParser) -> None:
        """Fill the argparse parser with the app general settings."""
        app_subgroup = parser.add_argument_group(title="App General Settings", description="General settings for the application.")
        app_subgroup.add_argument("--execute_sparql", action='store_true', help="Whether to execute SPARQL queries in the application.")   
        
    def set_from_args(self, args: argparse.Namespace) -> None:
        """Set the app general settings from the parsed command line arguments."""
        self.execute_sparql = getattr(args, "execute_sparql", self.execute_sparql)                                                                                                                                                                       
                                                                                                                                                                                                            
class EndpointConfiguration(BaseModel):                                                                                                                                                                       
    base_url: str                                                                                                                                                                                        
    repository: str                                                                                                                                                                     
    username: str|None = None                                                                                                                                                                       
    password: str|None = None      
    
    def get_full_endpoint(self) -> str:
        return f"{self.base_url}{self.repository}"                                                                                                                                                                 
                                                                                                                                                                                                            
class GraphConfiguration(BaseModel):                                                                                                                                                                          
    rdf_store_endpoint_configuration: str
    structural_index: str                                                                                                                                                                                
    search_index: str
    
    def get_full_endpoint(self) -> str:
        return ENDPOINT_CONFIGURATIONS[self.rdf_store_endpoint_configuration].get_full_endpoint()
    
    def get_endpoint_configuration(self) -> EndpointConfiguration:
        return ENDPOINT_CONFIGURATIONS[self.rdf_store_endpoint_configuration]
    
class RuntimeSettings(BaseModel):
    """Settings that are determined at runtime and not loaded from configuration files (includes setting overrides from command line arguments)."""
    # EndpointConfiguration & GraphConfiguration overrides.
    rdf_endpoint_url: str = ""
    rdf_endpoint_username: str = ""
    rdf_endpoint_password: str = ""
    structural_index: str = ""
    search_index: str = ""
    
    def endpoint_fill_parse_args(self, parser: argparse.ArgumentParser|argparse._ArgumentGroup):
        """Fill the argparse parser with the runtime settings."""
        if isinstance(parser, argparse.ArgumentParser):
            runtime_subgroup = parser.add_argument_group(title="Runtime RDF Endpoint Settings", description="Settings that can be overridden at runtime.")
        else:
            runtime_subgroup = parser
        runtime_subgroup.add_argument("--rdf_endpoint_url", type=str, required=False, help="Override for the RDF store endpoint URL.")
        runtime_subgroup.add_argument("--rdf_endpoint_username", type=str, required=False, help="Override for the RDF store endpoint username.")
        runtime_subgroup.add_argument("--rdf_endpoint_password", type=str, required=False, help="Override for the RDF store endpoint password.")
    
    def graph_fill_parse_args(self, parser: argparse.ArgumentParser) -> None:
        if isinstance(parser, argparse.ArgumentParser):
            runtime_subgroup = parser.add_argument_group(title="Runtime Knowledge Graph Settings", description="Settings that can be overridden at runtime.")
        else:
            runtime_subgroup = parser
        runtime_subgroup.add_argument("--structural_index", type=str, required=False, help="Override for the structural index name.")
        runtime_subgroup.add_argument("--search_index", type=str, required=False, help="Override for the search index name.")
    
    def set_from_args(self, args: argparse.Namespace) -> None:
        """Set the runtime settings from the parsed command line arguments."""
        self.rdf_endpoint_url = getattr(args, "rdf_endpoint_url", "")
        self.rdf_endpoint_username = getattr(args, "rdf_endpoint_username", "")
        self.rdf_endpoint_password = getattr(args, "rdf_endpoint_password", "")
        self.structural_index = getattr(args, "structural_index", "")
        self.search_index = getattr(args, "search_index", "")
    
# -------------------------------
# ----- LOAD CONFIGURATIONS -----
# -------------------------------
    
ENV_PATTERN = re.compile(r"^env\((?P<var>[^:]+):?(?P<default>[^)]*)\)$")                                                                                                                                 
                                                                                                                                                                                                             
def _env_constructor(loader, node):                                                                                                                                                                      
    val = loader.construct_scalar(node)                                                                                                                                                                  
    match = ENV_PATTERN.match(val)                                                                                                                                                                       
    if match:                                                                                                                                                                                            
        var = match.group("var")                                                                                                                                                                         
        default = match.group("default")                                                                                                                                                                 
        default_val = None if default.lower() == "null" else default                                                                                                                                             
        return os.getenv(var, default_val)
    return val                                                                                                                                                                                           
                                                                                                                                                                                                            
yaml.SafeLoader.add_implicit_resolver("!env", ENV_PATTERN, None)                                                                                                                                         
yaml.SafeLoader.add_constructor("!env", _env_constructor)                                                                                                                                                
                                                                                                                                                                                                            
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"       

print(f"Loading configuration files from: {CONFIG_DIR}")                                                                                                                                               
                                                                                                                                                                                                            
def load_yaml(filename: str) -> dict:                                                                                                                                                                    
    file_path = CONFIG_DIR / filename                                                                                                                                                                    
    if not file_path.exists():                                                                                                                                                                           
        return {}                                                                                                                                                                                        
    with open(file_path, "r", encoding="utf-8") as f:                                                                                                                                                    
        return yaml.safe_load(f) or {}
    
_general = load_yaml("core.yml")
_core = load_yaml("core.yml")
_app = load_yaml("app.yml")                                                                                                                                                                              
_endpoints = load_yaml("endpoints.yml")                                                                                                                                                                  
_graphs = load_yaml("graphs.yml")

GENERAL_SETTINGS = GeneralSettings(**_general)
AGENT_SETTINGS = AgentSettings(**_core["agent"])                                                                                                                                                                                                                                                                                                                       
LLM_SETTINGS = LlmSettings(**_core["llm"]) 
QUERY_DATABASE_SETTINGS = QueryDatabaseSettings(**_core["qd"])
GRAPH_SEARCH_SETTINGS = GraphSearchSettings(**_core["graph_search"])
APP_SETTINGS = AppGeneralSettings(**_app["general"])
                                                                                                                                                                                                             
ENDPOINT_CONFIGURATIONS: dict[str, EndpointConfiguration] = {                                                                                                                                                               
    name: EndpointConfiguration(**data)                                                                                                                                                                       
    for name, data in _endpoints.items()                                                                                                                                                                 
    if not name.startswith("_")  # Skips yaml anchor templates like _templates                                                                                                                           
}                                                                                                                                                                                                        
                                                                                                                                                                                                            
GRAPH_CONFIGURATIONS: dict[str, GraphConfiguration] = {                                                                                                                                                                     
    name: GraphConfiguration(**data)                                                                                                                                                                          
    for name, data in _graphs.items()                                                                                                                                                                    
}

RUNTIME_SETTINGS = RuntimeSettings()  # Initialize with default values; will be updated at runtime based on command line arguments.

def print_configurations():
    print("Agent Settings:", AGENT_SETTINGS)
    print("LLM Settings:", LLM_SETTINGS)
    print("Query Database Settings:", QUERY_DATABASE_SETTINGS)
    print("App Settings:", APP_SETTINGS)
    print("Endpoint Configurations:", ENDPOINT_CONFIGURATIONS)
    print("Graph Configurations:", GRAPH_CONFIGURATIONS)
    print("Runtime Settings:", RUNTIME_SETTINGS)
    
def get_configurations_dump() -> dict:
    return {
        "general": GENERAL_SETTINGS.model_dump(),
        "agent": AGENT_SETTINGS.model_dump(),
        "llm": LLM_SETTINGS.model_dump(),
        "query_database": QUERY_DATABASE_SETTINGS.model_dump(),
        "graph_search": GRAPH_SEARCH_SETTINGS.model_dump(),
        "app": APP_SETTINGS.model_dump(),
        "endpoints": {name: config.model_dump() for name, config in ENDPOINT_CONFIGURATIONS.items()},
        "graphs": {name: config.model_dump() for name, config in GRAPH_CONFIGURATIONS.items()},
        "runtime": RUNTIME_SETTINGS.model_dump()
    }
    
print('Configurations loaded...')

# ------------------------------
# ----- PYTHIA ARGS PARSER -----
# ------------------------------

import argparse

class PythiaArgParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        base_description = (
            "PYTHIA: Universal Question Answering over Knowledge Graphs.\n"
            "Developed by Sergios-Anestis Kefalidis\n"
            "v" + GENERAL_SETTINGS.version + "\n\n"
        )
        kwargs["description"] = base_description + (
            kwargs.get("description", "") or ""
        )
        # Preserve line breaks in description (and epilog)
        kwargs.setdefault(
            "formatter_class", argparse.RawDescriptionHelpFormatter
        )

        super().__init__(*args, **kwargs)
        
    @override
    def parse_args(self) -> argparse.Namespace:
        """Parse the command line arguments and set configurations accordingly."""
        args = super().parse_args()
        self._set_configs_from_args(args)
        return args
    
    def _set_configs_from_args(self, args: argparse.Namespace) -> None:
        """Set all configurations from the parsed command line arguments."""
        AGENT_SETTINGS.set_from_args(args)
        LLM_SETTINGS.set_from_args(args)
        QUERY_DATABASE_SETTINGS.set_from_args(args)
        APP_SETTINGS.set_from_args(args)
        RUNTIME_SETTINGS.set_from_args(args)
