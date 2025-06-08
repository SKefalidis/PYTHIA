import argparse
from llama_index.retrievers.bm25 import BM25Retriever



def prepare_retriever(args):
    # ----------------------
    # ----- Load index -----
    # ----------------------
    print("Loading index...")
    retriever = BM25Retriever.from_persist_dir(args.path)
    retriever.similarity_top_k = 10

    return retriever
    
if __name__ == "__main__":
    argparse.ArgumentParser(description="LlamaIndex BM25 Index Playground")
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the index")
    args = parser.parse_args()
    
    retriever = prepare_retriever(args)
    
    while True:
        user_input = input("Enter your query (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        else:
            print(f"User input: {user_input}")
            response = retriever.retrieve(user_input)
            for node in response:
                print(f"Node: {node}")