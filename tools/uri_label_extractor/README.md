# uri_label_extractor

Java utility that extracts labels for URIs from RDF data and writes a TSV for *PYTHIA* workflows.

## Build

```bash
mvn clean package
```

Typical fat JAR:

- `target/uri-label-extractor-1.0-SNAPSHOT-jar-with-dependencies.jar`

## Usage

```bash
java -jar target/uri-label-extractor-1.0-SNAPSHOT-jar-with-dependencies.jar \
  -i ~/freebase.nt \
  -o ~/extracts_freebase/uri_labels.tsv \
  -l http://www.w3.org/2000/01/rdf-schema#label \
  -t 16
```

## Arguments

- `-i, --input` (required): RDF file or directory.
- `-o, --output` (required): output TSV path.
- `-l, --labels`: comma-separated label predicate URIs.
- `-t, --threads`: parallel workers.
- `-en, --english-only`: keep only English labels.

## Output format

`<uri>\t<label>`
