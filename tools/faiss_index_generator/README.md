# faiss_index_generator

Dense retrieval index builders and playground scripts for *PYTHIA* based on FAISS.

**It is recommended that you use `tools/all_indices` instead of this.**

## Files

- `create_faiss_index.py`: flat FAISS index (`IndexFlatL2`).
- `create_faiss_index_ivff.py`: IVF-Flat variant.
- `create_faiss_index_ivfpq.py`: high-accuracy quantized IVF-PQ + OPQ variant.
- `resume_faiss_index_ivfpq.py`: resumes build from `embeddings.mmap`.
- `test_faiss_index.py`: interactive search over a built index.

## Typical usage

```bash
python create_faiss_index_ivfpq.py --input ~/extracts_freebase/classes_full.tsv --output ~/indices_freebase/classes_full_index_faiss
```

Or smaller datasets:

```bash
python create_faiss_index.py --input ~/extracts_freebase/classes_full.tsv --output ~/indices_freebase/classes_full_index_faiss
```

## Test an index

```bash
python test_faiss_index.py --path ~/indices_freebase/classes_full_index_faiss
```

## Resume IVFPQ build

```bash
python resume_faiss_index_ivfpq.py --output ~/indices_freebase/classes_full_index_faiss --embedding-dim 768
```

## Common outputs

- `faiss.index`
- `docs.txt`
- `metadata.json` (for IVFPQ/resume variants)
