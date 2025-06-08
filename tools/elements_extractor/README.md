# elements_extractor

Core Java extractor for *PYTHIA*. It parses RDF triples and exports class/entity/predicate TSV files.

**This is the first step in the index generation process.**

## Build

```bash
mvn clean package
```

## Usage Example

In this example, the complete RDF dump of Freebase exists in `~/freebase.nt`. 

To generate our indices, we want to extract information about classes, named entities (instances) and predicates in the dump. We also want to filter any instances or classes that don't have labels, since Freebase uses IRIs that are not human readable, so un-named nodes would not be useful for entity linking (the main purpose of our indices).

We run the following command:

```bash
java -jar target/elements-extractor-1.0-SNAPSHOT-jar-with-dependencies.jar \
  -i ~/freebase.nt \
  -o ~/extracts_freebase/ \
  -kg freebase \
  -l http://www.w3.org/2000/01/rdf-schema#label \
  -npl \
  -d http://rdf.freebase.com/ns/common.topic.description \
  -f -fc
```

##### Note: Similar commands are used for all knowledge graphs, with the necessary changes in label and description predicates (these are Freebase specific).

## Required arguments

- `-i, --input`: RDF input file or directory.
- `-o, --output`: output directory.
- `-kg, --knowledge_graph`: one of `wikidata`, `dbpedia`, `freebase`, `generic`.

Generic works for all knowledge graphs. For Wikidata, DBpedia and Freebase we support some minor cleanup (for Wikidata predicates are named differently when they appear as entities, so we link those, in Freebase we employ a targeted fix for our label-extraction logic, in DBpedia we avoid redirection entities). 

This does not violate the KG-agnostic and zero-shot nature of our architecture, we still answer questions over the entire KG without utilizing any special characteristics of the knowledge graph or any information from the dataset.

## Optional arguments

- `-l, --labels`: comma-separated label predicate URIs.
- `-d, --descriptions`: comma-separated description predicate URIs.
- `-c, --classes`: comma-separated class-defining predicate URIs.
- `-t, --threads`: number of threads.
- `-f, --filter`: filter entities without labels.
- `-fc, --filter-classes`: filter classes without labels.
- `-ep, --entity_prefixes`: comma-separated URI substrings for entity filtering.
- `-cp, --class_prefixes`: comma-separated URI substrings for class filtering.
- `-nel`, `-npl`, `-ncl`: derive entity/predicate/class labels from URIs.
- `-mem`, `-compref`: memory optimization options (experimental, prone to breaking).

## Generated files

- `entities.tsv`, `entities_labels.tsv`, `entities_full.tsv`
- `classes.tsv`, `classes_labels.tsv`, `classes_full.tsv`
- `predicates.tsv`, `predicates_labels.tsv`, `predicates_full.tsv`
- `all.tsv`, `all_with_types.tsv`, `uri_labels.tsv`