# llama_index_generator

LlamaIndex-based utilities for dense and BM25 retrieval index generation in *PYTHIA*.

**It is recommended that you use `tools/all_indices` instead of this.**

## Files

- `create_llama_index.py`: dense vector index.
- `create_llama_bm25_index.py`: BM25 index.
- `create_llama_bm25_index_alt.py`: batch-oriented BM25 build.
- `test_llama_index.py`: interactive dense retrieval test.
- `test_llama_bm25_index.py`: interactive BM25 retrieval test.

## Build dense index

```bash
python create_llama_index.py --input ~/extracts_freebase/classes_full.tsv --output ~/indices_freebase/classes_full_index_llama
```

## Build BM25 index

```bash
python create_llama_bm25_index.py --input ~/extracts_freebase/classes.tsv --output ~/indices_freebase/classes_index_bm25
```

For larger files, use:

```bash
python create_llama_bm25_index_alt.py --input ~/extracts_freebase/classes.tsv --output ~/indices_freebase/classes_index_bm25
```

## Interactive tests

```bash
python test_llama_index.py --path ~/indices_freebase/classes_full_index_llama
python test_llama_bm25_index.py --path ~/indices_freebase/classes_index_bm25
```
