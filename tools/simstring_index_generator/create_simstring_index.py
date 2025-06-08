import pickle
import argparse
import os
import sys
from hashlib import md5
from tqdm import tqdm
import simstring


def create_index(args):
    tqdm_kwargs = {"desc": "Reading lines", "file": sys.stdout}

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # ------------------------
    # ----- Prepare data -----
    # ------------------------
    print("Preparing data...")
    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        texts = f.read().split("\n")

    key_to_value = {}
    skipped = 0
    for line in tqdm(texts, **tqdm_kwargs):
        if not line:
            skipped += 1
            continue
        parts = line.split("\t", 1)
        parts = [x for x in parts if x != '']
        if len(parts) == 2:
            key = parts[1].strip().lower()  # Use the second part as key, and normalize to lowercase
            value = parts[0].strip()
        else:
            print(f"[Warning] Skipping line (does not have 2 tab-separated parts): {line}")
            continue
        if key in key_to_value:
            key_to_value[key].append(value)
        else:
            key_to_value[key] = [value]

    print(f"Loaded {len(key_to_value)} pairs (skipped {skipped})")

    # ---------------------
    # ----- simstring -----
    # ---------------------
    print(f"Creating simstring index with sharding: {args.shards} shards...")

    # Deterministic shard assignment using md5(key) % shards
    def shard_id_for(key: str) -> int:
        return int(md5(key.encode("utf-8")).hexdigest(), 16) % args.shards

    # Pre-create shard directories
    shard_dirs = []
    for i in range(args.shards):
        shard_dir = os.path.join(args.output, f"shard_{i:02d}")
        os.makedirs(shard_dir, exist_ok=True)
        shard_dirs.append(shard_dir)

    # Build one DB per shard
    shard_counts = [0] * args.shards
    writers = {}
    try:
        for i, shard_dir in enumerate(shard_dirs):
            db_path = os.path.join(shard_dir, "keys.db")
            # Set n-gram and marking via constructor; do not set attributes later
            writers[i] = simstring.writer(db_path, n=3, be=False, unicode=False)

        for key in tqdm(key_to_value.keys(), desc="Indexing keys", file=sys.stdout):
            sid = shard_id_for(key)
            writers[sid].insert(key)
            shard_counts[sid] += 1
    finally:
        # Ensure all writers are closed even if an exception occurs
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass

    # Map keys to values at root
    pkl_path = os.path.join(args.output, "key_to_value.pkl")
    with open(pkl_path, "wb") as pf:
        pickle.dump(key_to_value, pf)

    # Save a small metadata file about sharding
    meta_path = os.path.join(args.output, "shards.meta")
    try:
        with open(meta_path, "w", encoding="utf-8") as mf:
            mf.write(f"shards={args.shards}\n")
            for i, c in enumerate(shard_counts):
                mf.write(f"shard_{i:02d}={c}\n")
    except Exception:
        # Non-fatal
        pass

    print("Index generation complete!")
    print(f"Shards created under: {args.output}")
    for i, c in enumerate(shard_counts):
        print(f"  - shard_{i:02d}: {c} keys")
    print(f"Mapping saved to: {pkl_path}")
    

def populate_parser_args(parser):
    parser.add_argument("--input", type=str, required=True, help="Path to the input TSV file (key\tvalue per line)")
    parser.add_argument("--output", type=str, required=True, help="Directory to write the simstring DB and mapping")
    parser.add_argument("--shards", type=int, default=10, help="Number of shards to split the DB into (default: 10)")
    return parser

def get_parser():
    parser = argparse.ArgumentParser(description="SimString Index Generator")
    return populate_parser_args(parser)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    create_index(args)
