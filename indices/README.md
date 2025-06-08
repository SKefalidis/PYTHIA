## Default Index Directory

This is the default directory used by *PYTHIA* to store knowledge graph indices.

### Setup Instructions
You should unpack your **custom indices** or **pre-built indices** into this directory. If you prefer to store them elsewhere, you must update the path in your `config.yaml` file.

### Directory Structure
The following structure is expected relative to the project root ($P):

```text
$P/indices/
└── freebase/
    ├── classes_full_index_faiss/
    ├── classes_labels_index_simstring/
    ├── entities_labels_index_simstring/
    ├── all.tsv
    ├── classes_predicates_no_literals.tsv
    └── classes_predicates.tsv
└── wikidata/
    ...
```

### Available Pre-built Indices

Pre-built indices are currently available for:

- [BESTIARY](https://figshare.com/s/e7d594a86683f20bfdf2)
- [DBpedia (October 2016)](https://figshare.com/s/e7d594a86683f20bfdf2)
- [Freebase](https://figshare.com/s/e7d594a86683f20bfdf2)

*Note: Availability is limited due to Figshare size constraints (used for anonymous review).*