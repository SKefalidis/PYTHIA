import argparse
from src.datasets.dataset import DatasetFactory
from src.elelem.provider import ProviderFactory
from src.utils import LogLevel, endpoints_fill_parse_args
from src.logging import create_logger
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs


GENERAL_PREFIXES = """
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX strdf: <http://strdf.di.uoa.gr/ontology#>
PREFIX uom: <http://www.opengis.net/def/uom/OGC/1.0/>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


def populate_parser_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "kg",
        type=str,
        choices=[kg.name for kg in KnowledgeGraphs],
        help="Target knowledge graph."
    )
    
    parser.add_argument(
        "kg_index_name",
        type=str,
        help="Name of the index to use for the selected knowledge graph."
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of dataset to use for the query database (if any).",
        choices=DatasetFactory.list_datasets()
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the database file.",
        default=None
    )
    
    parser.add_argument(
        "--pythia_system",
        type=str,
        help="Which version of Pythia to use.",
        choices=["pipeline", "agent-limited", "agentic"],
    )
    
    parser.add_argument(
        "--pythia_execute_sparql",
        type=bool,
        help="Whether to execute SPARQL queries against the endpoint instead of just returning them.",
    )
    
    ProviderFactory.fill_parse_args(parser)
    endpoints_fill_parse_args(parser)
    
    return parser


def init_from_args(args):
    target_kg = KnowledgeGraphs[args.kg]
    create_logger("pythia-main", './', log_level=LogLevel.INFO)
    if args.pythia_system == "pipeline":
        from src.engine.pipeline.pythia import Pythia
        pythia = Pythia(target_kg, args.kg_index_name, args.dataset_name, args.dataset_path)
    elif args.pythia_system == "agent-limited":
        from src.engine.agent.agent_limited import AgentLimited
        pythia = AgentLimited(target_kg, args.kg_index_name, args.dataset_name, args.dataset_path)
    elif args.pythia_system == "agentic":
        from src.engine.agent.agent_tools import AgentWithTools
        pythia = AgentWithTools(target_kg, args.kg_index_name, args.dataset_name, args.dataset_path)
    else:
        raise ValueError(f"Unknown Pythia system: {args.pythia_system}")
    return pythia