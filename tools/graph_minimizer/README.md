# graph_minimizer

Implementation of graph path search via [graph-tool](https://graph-tool.skewed.de/). In the default setup of *PYTHIA* this is used for the extended path search (i.e., not shortest paths, but the secondary search). If not using GraphDB this can be used to implement both search types.


## Usage

### 1) Encode RDF graph (Java)

Build:

```bash
mvn clean package
```

Run encoder:

```bash
java -jar target/graph-minimizer-1.0-SNAPSHOT-jar-with-dependencies.jar <input_nt_or_dir> <output_dir>
```

Outputs:

- `graph_encoded.txt`
- `entity_mapping.tsv`

### Intermediate Step - Environment Setup

At this point you need to setup the Python environment for `graph-tool`. Conda is recommended and an environment file is provided (`graph_minimizer.yml`). Because `graph-tool` uses a complex setup, if installation of the environment fail you might need to install it manually, following the official instructions.

#### Note: We recommend that you install the dependencies for this tooling in a separate environment than the main Pythia environment.

### 2) Build SQLite mapping index

```bash
python build_index.py <output_dir>/entity_mapping.tsv
```

Output:

- `<output_dir>/mapping.db`

### 3) Serve graph search API

```bash
python query_graph.py --source-dir <output_dir> --host 0.0.0.0 --port 65023
```

#### Note: This is the only step that is required if the indices are already built.

####  Required files in `--source-dir`
- `graph_encoded.txt`
- `mapping.db`
- optional: `classes.txt` (class filtering)
