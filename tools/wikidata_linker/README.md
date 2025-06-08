# wikidata_linker

Wikidata-specific post-processor for *PYTHIA* extractor outputs.

It moves property rows from `entities_*` files (e.g., `http://www.wikidata.org/entity/P31`) into predicate files (`/prop/direct/P31` and `/prop/direct-normalized/P31`) when matching predicate URIs already exist.

## Build

```bash
mvn clean package
```

Typical fat JAR:

- `target/wikidata-linker-1.0-SNAPSHOT-jar-with-dependencies.jar`

## Usage

```bash
java -jar target/wikidata-linker-1.0-SNAPSHOT-jar-with-dependencies.jar -d ~/extracts_wikidata
```

## Arguments

- `-d, --dir` (required): directory containing extractor TSV files.

## Expected files in directory

- `entities_labels.tsv`
- `entities_full.tsv`
- `predicates_labels.tsv` (created if missing)
- `predicates_full.tsv` (created if missing)

## Effect

- Rewrites entity/predicate TSV files in place.
- Keeps labels/descriptions and removes moved property rows from entity files.
