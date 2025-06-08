import argparse
from src.engine.config import CONFIG
from src.app.app import init_from_args, populate_parser_args, GENERAL_PREFIXES
from src.utils import execute_sparql_query, return_sparql_query_results


def run_cli(pythia):
    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        sparql_query = pythia.answer(question)
        sparql_query = GENERAL_PREFIXES + sparql_query
        print("Generated SPARQL Query:")
        print(sparql_query)
        
        if CONFIG().get('pythia_execute_sparql') == True:
            endpoint = pythia.kg.endpoint
            if endpoint:
                print("Executing SPARQL query against endpoint:", endpoint)
                try:
                    results = return_sparql_query_results(sparql_query, endpoint)
                    print("SPARQL Result:")
                    for row in results:
                        print('\t'.join(row))

                except Exception as e:
                    print(f"Error executing SPARQL: {e}")
            else:
                print("No endpoint URL configured; skipping execution.")
        # else:
        #     print("SPARQL execution not enabled; skipping execution.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for Pythia.")
    parser = populate_parser_args(parser)
    args = parser.parse_args()
    pythia = init_from_args(args)
    
    run_cli(pythia)
