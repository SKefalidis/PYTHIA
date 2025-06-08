# simstring_index_generator

Lexical retrieval index utilities for *PYTHIA* using SimString with optional sharding.

**It is recommended that you use `tools/all_indices` instead of this.**

## Files

- `create_simstring_index.py`: builds SimString DB(s) plus key-to-value mapping.
- `test_simstring_index.py`: interactive retrieval and ranking playground.

## Build index

```bash
python create_simstring_index.py \
  --input ~/extracts_freebase/entities_labels.tsv \
  --output ~/indices_freebase/entities_labels_index_simstring \
  --shards 10
```

## Test index

```bash
python test_simstring_index.py --path ~/indices_freebase/entities_labels_index_simstring
```

## Expected outputs

- `key_to_value.pkl`
- `shards.meta`
- either:
  - `keys.db` (single DB), or
  - `shard_XX/keys.db` (sharded layout)
