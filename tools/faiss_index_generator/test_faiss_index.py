import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

def load_faiss_index(index_dir):
    print("Loading FAISS index...")
    index = faiss.read_index(index_dir + "/faiss.index")

    print("Loading documents...")
    with open(index_dir + '/docs.txt', "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines()]
    
    return index, documents

def search_faiss_index(index, documents, query, k=30):
    print("Searching FAISS index...")
    query_vector = model.encode(query, prompt_name="query")
    query_vector = np.array(query_vector).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        results.append(documents[idx])
        print(f"[Distance: {dist:.4f}] {documents[idx]}")
    
    return results
    
if __name__ == "__main__":
    argparse.ArgumentParser(description="FAISS Dense Index Playground")
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the index")
    args = parser.parse_args()
    
    index, documents = load_faiss_index(args.path)
    
    while True:
        user_input = input("Enter your query (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        else:
            print(f"User input: {user_input}")
            response = search_faiss_index(index, documents, user_input)
            print(response)