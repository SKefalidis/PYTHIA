import torch
import argparse
import sys
from tqdm import tqdm
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
from datetime import datetime
import math


def create_faiss_index(args):
    tqdm_kwargs = {"desc": "Creating documents", "file": sys.stdout}

    # ------------------------
    # ----- Prepare data -----
    # ------------------------
    print("Preparing data...")
    with open(args.input, "r", encoding="utf-8") as f:
        raw_data = f.read()

    texts = raw_data.strip().split("\n")
    documents = [text for text in tqdm(texts, **tqdm_kwargs) if text.strip()]
    print(f"Loaded {len(documents):,} documents")

    # --------------------------
    # ----- Generate embeddings -----
    # --------------------------
    print("Generating embeddings (streaming to disk)...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "nomic-ai/nomic-embed-text-v2-moe"
    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        device=device,
    )

    batch_size = 256
    embedding_dim = model.get_sentence_embedding_dimension()
    num_docs = len(documents)

    # Create a memory-mapped file on disk (saves RAM)
    embed_path = os.path.join(args.output, "embeddings.mmap")
    embeddings_mmap = np.memmap(embed_path, dtype="float32", mode="w+", shape=(num_docs, embedding_dim))

    for i in tqdm(range(0, num_docs, batch_size), desc="Encoding batches", file=sys.stdout):
        batch = documents[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            show_progress_bar=False,
            batch_size=batch_size,
            device=device,
            prompt_name="passage",
        ).astype("float32")

        embeddings_mmap[i:i + len(batch_embeddings)] = batch_embeddings

    # Flush to disk
    print("Flushing embeddings to disk...")
    embeddings_mmap.flush()
    del embeddings_mmap
    import gc, time
    gc.collect()
    time.sleep(2)

    # Verify file size
    expected_size = num_docs * embedding_dim * 4
    actual_size = os.path.getsize(embed_path)
    if actual_size != expected_size:
        raise RuntimeError(f"Embedding file incomplete! Expected {expected_size}, got {actual_size}")

    # Reopen safely
    embeddings = np.memmap(embed_path, dtype="float32", mode="r+", shape=(num_docs, embedding_dim))
    print(f"Embeddings successfully remapped: shape={embeddings.shape}")

    # --------------------------
    # ----- Normalize (cosine) -----
    # --------------------------
    # Downstream search normalizes the query vector when using IVFPQ/OPQ.
    # To ensure cosine-equivalent L2 distances, normalize database vectors too.
    print("Normalizing embeddings (L2) for cosine similarity...")
    faiss.normalize_L2(embeddings)

    # --------------------------
    # ----- Create FAISS index -----
    # --------------------------
    print("Creating FAISS index (high-accuracy quantization)...")
    dimension = embeddings.shape[1]

    # ----- Parameters -----
    N = len(embeddings)
    # Adaptive nlist based on dataset size (bounded)
    nlist = min(65536, max(32, int(4 * math.sqrt(max(1, N)))))
    nlist = min(nlist, max(1, N))
    m = 32              # desired number of subquantizers
    nbits = 8           # bits per subquantizer
    use_opq = True      # rotation matrix for better accuracy

    # Ensure m divides dimension; pick the largest divisor <= 64 (or <= dimension)
    if dimension % m != 0:
        print(f"Warning: m={m} does not divide dimension d={dimension}. Choosing a compatible m...")
        # candidates: all divisors of d up to 64 (or d if smaller), prefer larger for better recall
        max_m = min(64, dimension)
        divisors = [k for k in range(max_m, 0, -1) if dimension % k == 0]
        # Heuristic: prefer multiples of 8 or 16 when possible for efficiency
        preferred = [k for k in divisors if k % 16 == 0] or [k for k in divisors if k % 8 == 0] or divisors
        m = preferred[0]
        print(f"Selected m={m} (d/m={dimension//m})")

    quantizer = faiss.IndexFlatL2(dimension)

    if use_opq:
        print("Using OPQ (Optimized Product Quantization) for higher accuracy...")
        opq = faiss.OPQMatrix(dimension, m)
        ivfpq = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
        index = faiss.IndexPreTransform(opq, ivfpq)
    else:
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)

    # --------------------------
    # ----- Train the index -----
    # --------------------------
    print("Training FAISS index...")
    # Use a subset for training (e.g., up to 100k vectors)
    np.random.seed(42)
    n_train = min(len(embeddings), 100000)
    # Ensure we have at least as many training points as nlist
    if n_train < nlist:
        n_train = min(N, max(nlist, 2 * nlist))
    random_idx = np.random.choice(len(embeddings), n_train, replace=False)
    train_data = embeddings[random_idx]

    index.train(train_data)
    print("Training complete!")

    # --------------------------
    # ----- Add embeddings -----
    # --------------------------
    print("Adding embeddings to index (in batches)...")
    add_batch = 100_000
    total_added = 0
    for start in tqdm(range(0, N, add_batch), desc="Adding batches", file=sys.stdout):
        end = min(start + add_batch, N)
        index.add(embeddings[start:end])
        total_added += (end - start)
    print(f"Total indexed vectors: {index.ntotal:,}")

    # --------------------------
    # ----- Save the index -----
    # --------------------------
    print("Saving index to disk...")
    os.makedirs(args.output, exist_ok=True)
    index_path = os.path.join(args.output, "faiss.index")
    faiss.write_index(index, index_path)

    # Save associated documents
    docs_path = os.path.join(args.output, "docs.txt")
    with open(docs_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc + "\n")

    # Save metadata JSON
    meta = {
        "model": model_name,
        "device": device,
        "dimension": int(dimension),
        "nlist": int(nlist),
        "m": int(m),
        "nbits": int(nbits),
        "use_opq": bool(use_opq),
        "normalized_l2": True,
        "num_vectors": int(N),
        "train_size": int(n_train),
        "batch_size_encode": int(batch_size),
        "batch_size_add": int(add_batch),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    meta_path = os.path.join(args.output, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Print summary with improved memory estimate (codes only)
    codes_bytes = N * (m * nbits // 8)
    print("Index generation complete!")
    print(f"FAISS index saved to: {index_path}")
    print(f"Documents saved to: {docs_path}")
    print(f"Metadata saved to: {meta_path}")
    print("\n--- Summary ---")
    print(f"Index type: IVFPQ (m={m}, nbits={nbits}, nlist={nlist}, OPQ={use_opq})")
    print(f"Approx. codes memory: ~{codes_bytes:,} bytes ≈ {codes_bytes / (1024**3):.2f} GB")
    print("(Excludes coarse centroids and OPQ overhead)")
    print("----------------")
    
    if os.path.exists(embed_path):
        os.remove(embed_path)
        print(f"Deleted temporary embeddings file: {embed_path}")


def populate_parser_args(parser):
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAISS High-Accuracy Quantized Index Generator")
    parser = populate_parser_args(parser)
    args = parser.parse_args()

    create_faiss_index(args)
