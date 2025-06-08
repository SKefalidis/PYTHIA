#!/usr/bin/env python3
import os
import sys
import math
import json
import time
import gc
import numpy as np
import faiss
from tqdm import tqdm
from datetime import datetime
import argparse


def resume_faiss_index(args):
    print("=== FAISS Resume Script ===")

    output_dir = args.output
    embed_path = os.path.join(output_dir, "embeddings.mmap")
    docs_path = os.path.join(output_dir, "docs.txt")

    if not os.path.exists(embed_path):
        raise FileNotFoundError(f"Missing embeddings.mmap at {embed_path}")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Missing docs.txt at {docs_path}")

    print("Loading documents...")
    with open(docs_path, "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]
    num_docs = len(documents)
    print(f"Loaded {num_docs:,} documents")

    # Verify mmap integrity
    embedding_dim = args.embedding_dim
    expected_bytes = num_docs * embedding_dim * 4
    actual_bytes = os.path.getsize(embed_path)
    print(f"Expected: {expected_bytes:,} bytes, actual: {actual_bytes:,} bytes")
    if expected_bytes != actual_bytes:
        raise RuntimeError("❌ embeddings.mmap file appears incomplete!")

    print("Remapping embeddings...")
    time.sleep(1)
    embeddings = np.memmap(embed_path, dtype="float32", mode="r+", shape=(num_docs, embedding_dim))
    print(f"✅ mmap reopened, shape={embeddings.shape}")

    # Normalize embeddings safely
    print("Normalizing embeddings (L2)...")
    for i in tqdm(range(0, num_docs, 100_000), desc="Normalizing", file=sys.stdout):
        faiss.normalize_L2(embeddings[i:i + 100_000])
    print("✅ Normalization complete.")

    # Build FAISS index
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    N = num_docs
    nlist = min(65536, max(32, int(4 * math.sqrt(N))))
    m, nbits, use_opq = 32, 8, True

    # Ensure m divides dimension
    if dimension % m != 0:
        max_m = min(64, dimension)
        divisors = [k for k in range(max_m, 0, -1) if dimension % k == 0]
        preferred = [k for k in divisors if k % 16 == 0] or [k for k in divisors if k % 8 == 0] or divisors
        m = preferred[0]
        print(f"Adjusted m to {m}")

    quantizer = faiss.IndexFlatL2(dimension)
    if use_opq:
        opq = faiss.OPQMatrix(dimension, m)
        ivfpq = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
        index = faiss.IndexPreTransform(opq, ivfpq)
    else:
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)

    print("Training FAISS index...")
    np.random.seed(42)
    n_train = min(len(embeddings), 100_000)
    random_idx = np.random.choice(len(embeddings), n_train, replace=False)
    index.train(embeddings[random_idx])
    print("✅ Training complete.")

    print("Adding embeddings to index...")
    for start in tqdm(range(0, N, 100_000), desc="Adding", file=sys.stdout):
        end = min(start + 100_000, N)
        index.add(embeddings[start:end])
    print(f"✅ Added {index.ntotal:,} vectors")

    # Save results
    print("Saving index and metadata...")
    index_path = os.path.join(output_dir, "faiss.index")
    faiss.write_index(index, index_path)

    meta = {
        "dimension": int(dimension),
        "nlist": int(nlist),
        "m": int(m),
        "nbits": int(nbits),
        "use_opq": bool(use_opq),
        "num_vectors": int(N),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ FAISS index successfully built!")
    print(f"Index path: {index_path}")
    print(f"Metadata:  {os.path.join(output_dir, 'metadata.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume FAISS index building from existing embeddings.mmap")
    parser.add_argument("--output", type=str, required=True, help="Output directory containing embeddings.mmap and docs.txt")
    parser.add_argument("--embedding-dim", type=int, default=768, help="Embedding dimension (default: 768)")
    args = parser.parse_args()
    resume_faiss_index(args)
