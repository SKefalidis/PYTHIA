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
    with open(args.input, "r", encoding="utf-8") as f:
        raw_data = f.read()
    
    texts = raw_data.split("\n")
    documents = [text for text in tqdm(texts, **tqdm_kwargs)]

    # --------------------------
    # ----- Generate embeddings -----
    # --------------------------
    print("Generating embeddings...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True, device=device)

    batch_size = 256  # Reduce batch size to avoid memory overflow
    embeddings_list = []
    
    # Generate and store embeddings in batches
    for i in tqdm(range(0, len(documents), batch_size), desc="Processing batches", file=sys.stdout):
        batch = documents[i:i + batch_size]
        embeddings = model.encode(
            batch, 
            show_progress_bar=False, 
            batch_size=batch_size,
            device=device,
            prompt_name="passage"
        )
        embeddings_list.append(embeddings)
    
    embeddings = np.vstack(embeddings_list).astype("float32")  # Stack all embeddings

    # --------------------------
    # ----- Create FAISS index -----
    # --------------------------
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    
    # Use IVF index for more memory-efficient indexing
    quantizer = faiss.IndexFlatL2(dimension)  # The quantizer is typically a flat index
    index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 is the number of clusters (adjustable)

    # Train the index (required for IVF indexing)
    print("Training FAISS index...")
    index.train(embeddings)  # This step trains the quantizer

    # Add embeddings to the index
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
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory")
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAISS Dense Index Generator")
    parser = populate_parser_args(parser)
    args = parser.parse_args()

    create_faiss_index(args)
