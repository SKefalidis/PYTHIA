import argparse
import os
import sys
import subprocess
from typing import List


FAISS_FILES = {
    "predicates_full.tsv", 
    # "entities_full.tsv", 
    "classes_full.tsv"
}
BM25_FILES = {
    # "predicates_labels.tsv",
    # "entities_labels.tsv",
    # "classes_labels.tsv",
    # "predicates.tsv",
    # "entities.tsv",
    # "classes.tsv",
}
SIMSTRING_FILES = {
    "predicates_labels.tsv",
    "entities_labels.tsv",
    "classes_labels.tsv",
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate indices from Elements Extractor TSV outputs. "
            "FAISS for predicates_full.tsv, entities_full.tsv, classes_full.tsv; BM25 for the rest."
        )
    )
    parser.add_argument("--input-dir", required=True, type=str, help="Directory with TSV files")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to store generated indices")

    parser.add_argument(
        "--mode",
        choices=["all", "bm25", "faiss", "simstring"],
        default="all",
        help="Which index type(s) to generate: all (default), bm25, faiss, or simstring."
    )

    parser.add_argument("--shards", type=int, default=10, help="Number of shards to split the DB into (default: 10)")
    
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)
        
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Resolve helper scripts
    tools_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    faiss_script_ivfpq = os.path.join(tools_root, "faiss_index_generator", "create_faiss_index_ivfpq.py")
    faiss_script_flat = os.path.join(tools_root, "faiss_index_generator", "create_faiss_index.py")
    bm25_script = os.path.join(tools_root, "llama_index_generator", "create_llama_bm25_index_alt.py")
    simstring_script = os.path.join(tools_root, "simstring_index_generator", "create_simstring_index.py")

    if (
        not os.path.isfile(faiss_script_ivfpq)
        or not os.path.isfile(faiss_script_flat)
        or not os.path.isfile(bm25_script)
        or not os.path.isfile(simstring_script)
    ):
        print("Missing helper scripts. Expected:")
        print(" -", faiss_script_ivfpq)
        print(" -", faiss_script_flat)
        print(" -", bm25_script)
        print(" -", simstring_script)
        sys.exit(2)

    # Iterate TSV files
    entries = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".tsv")])
    if not entries:
        print("No TSV files found.")
        return

    def count_non_empty_lines(path: str, limit: int | None = None) -> int:
        """Count non-empty lines in a file efficiently.
        If limit is provided, stop counting once the count exceeds limit.
        """
        count = 0
        # Larger buffer reduces syscalls on big files
        with open(path, "r", encoding="utf-8", errors="ignore", buffering=1024 * 1024) as fh:
            for line in fh:
                # Avoid creating a new string via strip(); issapce() is faster and allocation-free
                if line and not line.isspace():
                    count += 1
                    if limit is not None and count > limit:
                        break
        return count

    FAISS_IVFPQ_THRESHOLD = 1_000_000


    for filename in entries:
        file_path = os.path.join(input_dir, filename)
        stem = os.path.splitext(filename)[0]

        cmds: List[List[str]] = []

        # Only run the selected mode(s)
        if args.mode in ("all", "faiss") and filename in FAISS_FILES:
            try:
                num_lines = count_non_empty_lines(file_path, limit=FAISS_IVFPQ_THRESHOLD)
            except Exception as e:
                print(f"Warning: failed to count lines in {file_path}: {e}. Defaulting to flat FAISS.")
                num_lines = 0

            use_ivfpq = num_lines > FAISS_IVFPQ_THRESHOLD
            out_dir = os.path.join(output_dir, stem + "_index_faiss")
            try:
                os.makedirs(out_dir, exist_ok=False)
                if use_ivfpq:
                    print(f"\n[FAISS-IVFPQ] {filename} (>" + f"{FAISS_IVFPQ_THRESHOLD:,}" + f" lines) -> {out_dir}")
                    cmds.append([sys.executable, faiss_script_ivfpq, "--input", file_path, "--output", out_dir])
                else:
                    print(f"\n[FAISS-FLAT]  {filename} (<= {FAISS_IVFPQ_THRESHOLD:,} lines) -> {out_dir}")
                    cmds.append([sys.executable, faiss_script_flat, "--input", file_path, "--output", out_dir])
            except FileExistsError:
                print(f"Directory already exists: {out_dir}")

        if args.mode in ("all", "bm25") and filename in BM25_FILES:
            out_dir = os.path.join(output_dir, stem + "_index_bm25")
            try:
                os.makedirs(out_dir, exist_ok=False)
                print(f"\n[BM25]        {filename} -> {out_dir}")
                cmds.append([sys.executable, bm25_script, "--input", file_path, "--output", out_dir])
            except FileExistsError:
                print(f"Directory already exists: {out_dir}")

        if args.mode in ("all", "simstring") and filename in SIMSTRING_FILES:
            out_dir = os.path.join(output_dir, stem + "_index_simstring")
            try:
                os.makedirs(out_dir, exist_ok=False)
                print(f"\n[SimString]   {filename} -> {out_dir}")
                cmds.append([sys.executable, simstring_script, "--input", file_path, "--output", out_dir, "--shards", str(args.shards)])
            except FileExistsError:
                print(f"Directory already exists: {out_dir}")

        # Execute all commands for this file
        for cmd in cmds:
            code = subprocess.call(cmd)
            if code != 0:
                print(f"Command failed with exit code {code}: {' '.join(cmd)}")
                sys.exit(code)


if __name__ == "__main__":    
    main()

