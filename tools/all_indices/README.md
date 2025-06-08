# all_indices

Index generation utility. This script scans extractor TSV outputs (from `tools/elements_extractor/`) and dispatches index builders for supported files.

## What it does

- Reads all `.tsv` files from `--input-dir`.
- Generates selected index types in `--output-dir`.
- Uses file-name rules to decide which indexers to call.

Current defaults in the script:

- FAISS: `classes_full.tsv`, `predicates_full.tsv`
- SimString: `classes_labels.tsv`, `entities_labels.tsv`, `predicates_labels.tsv`
- BM25: Currently disabled to save time since it is not used, can be re-enabled by editing `generator.py`.

## Usage Example

```bash
python generator.py \
  --input-dir ~/extracts_freebase \
  --output-dir ~/indices_freebase \
  --mode all \
  --shards 10
```

## Arguments

- `--input-dir` (required): folder with TSV files.
- `--output-dir` (required): folder where index directories are created.
- `--mode`: `all`, `bm25`, `faiss`, `simstring` (default: `all`).
- `--shards`: shard count for SimString generation (default: `10`). Shards are used because simstring is limited 32-bit address spaces.
