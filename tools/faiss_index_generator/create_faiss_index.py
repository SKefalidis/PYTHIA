import torch
import argparse
import sys
from tqdm import tqdm
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def create_faiss_index(args):
    tqdm_kwargs = {"desc": "Creating documents", "file": sys.stdout}
    
    # ------------------------
    # ----- Prepare data -----
    # ------------------------
    print("Preparing data...")
    raw_data = open(args.input).read()
    texts = raw_data.split("\n")
    documents = [text for text in tqdm(texts, **tqdm_kwargs)] # need prefix, could also add this via SentenceTransformer
    documents_texts = [" ".join(text.split('\t')[1:]) for text in documents if text.strip() != ""]

    # --------------------------
    # ----- Generate embeddings -----
    # --------------------------
    print("Generating embeddings...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True, device=device)
    
    embeddings = model.encode(
        documents_texts, 
        show_progress_bar=True, 
        batch_size=256,
        device=device,
        prompt_name="passage"
    )

    embeddings = np.array(embeddings).astype("float32")  # FAISS requires float32

    # Normalize embeddings for cosine similarity (L2 norm = 1)
    faiss.normalize_L2(embeddings)

    # --------------------------
    # ----- Create FAISS index -----
    # --------------------------
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # ----------------------------------
    # ----- Save the index to disk -----
    # ----------------------------------
    print("Saving index to disk...")
    os.makedirs(args.output, exist_ok=True)
    faiss.write_index(index, f"{args.output}/faiss.index")
    
    with open(f"{args.output}/docs.txt", "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc + "\n")

    print("Index generation complete!")
    print(f"Index and documents saved to {args.output}")
    

def populate_parser_args(parser):
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output index file")
    return parser

def get_parser():
    parser = argparse.ArgumentParser(description="FAISS Dense Index Generator")
    return populate_parser_args(parser)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    create_faiss_index(args)
