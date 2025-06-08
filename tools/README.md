# *PYTHIA* Tools

This folder contains utility scripts and standalone services used to prepare data, build indices, and run support components for *PYTHIA* .

## Contents

- `all_indices/`: orchestrates index generation from extractor outputs.
- `bela_endpoint/`: FastAPI service exposing BELA entity linking.
- `class_predicates_extractor/`: Java extractor for class-predicate statistics.
- `elements_extractor/`: Java extractor for classes, entities, predicates TSVs (predicate indices are not used in the system currently).
- `extract_nl_answers/`: converts natural-language answers into CSV-like outputs via LLM.
- `faiss_index_generator/`: dense embedding index generation and tests.
- `file_splitter/`: simple shell script for splitting large files.
- `gost/`: packaged GoST JAR utility. GoST is a utility for parsing and modifying SPARQL queries.
- `graph_minimizer/`: alternative to GraphDB's path search functionality.
- `graphdb_server/`: helper service to control GraphDB and load repositories.
- `llama_index_generator/`: LlamaIndex dense/BM25 index generation and tests (unused).
- `simstring_index_generator/`: SimString lexical index generation and tests.
- `uri_label_extractor/`: Java utility to extract URI labels.
- `wikidata_linker/`: Java post-processor for Wikidata property rows.

## Pipeline

1. Run `elements_extractor` to produce `*_full.tsv`, `*_labels.tsv`, and base TSV files.
2. Optionally run `wikidata_linker` (Wikidata-only) to normalize property rows. We provide pre-built indices for Wikidata but because of Zenodo space limitations for the anonymous review these are not available.
3. Run `all_indices/generator.py` to build FAISS/SimString indices. We also support BM25 indices.
4. Run `class_predicates_extractor` for class-predicate statistics.
5. Move extractor outputs and index files to the appropriate index directory as described in the main `README.md` file.

## Conventions

- Most Python tools are intended to run from their own folder or from the project root with full paths.
- Java tools are Maven projects; build with:

```bash
mvn clean package
```

- Java fat JAR outputs are typically in `target/*-jar-with-dependencies.jar`.