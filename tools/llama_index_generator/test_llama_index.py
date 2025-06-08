import argparse
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def prepare_retriever(args):
    # ----------------------
    # ----- Load index -----
    # ----------------------
    print("Loading index...")
    embed_model = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
    storage_context = StorageContext.from_defaults(persist_dir=args.path)
    vector_index = load_index_from_storage(storage_context, embed_model=embed_model)
    retriever = VectorIndexRetriever(index=vector_index,
                                     similarity_top_k=5)   

    return retriever
    
if __name__ == "__main__":
    argparse.ArgumentParser(description="LlamaIndex Dense Index Playground")
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
            response = retriever.retrieve("search_query: " + user_input)
            for node in response:
                print(f"Node: {node}")