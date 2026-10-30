<div align="center">
   <img src="pythia_icon.png" alt="PYTHIA Icon" class="center" width="120"> 
</div>


<div align="center">
  <h1 style="font-style: italic;">PYTHIA</h1>
</div>

<div align="center" style="font-style: italic;">
  A Knowledge Graph Agnostic Question Answering System
</div>

## Overview
*PYTHIA* is a knowledge graph question-answering (**KGQA**) engine that is designed to function as a **plug-and-play** solution for general-purpose and specialized knowledge graphs (**KG**), without requiring any finetuning or modification.

## ✨ Highlights
- **Knowledge Graph Agnostic.** *PYTHIA* is a plug-and-play solution that can easily be deployed over any Knowledge Graph.
- **Zero-shot.** *PYTHIA* does not require any finetuning or training data to function.
- **State of the Art performance.** *PYTHIA* achieves new strong performance in popular question answering benchmarks, surpassing previous zero-shot solutions and closing the gap to finetuned systems.

## 🎉 News & Updates

- **[Aug 12, 2026]** *PYTHIA* [oral](https://github.com/SKefalidis/PYTHIA-dev/blob/main/docs/kdd-2026-presentation-material/PYTHIA-SLIDES-KDD-2026.pptx.pdf) presentation at [KDD '26](https://kdd2026.kdd.org/) !
- **[Aug 11, 2026]** *PYTHIA* [poster](https://github.com/SKefalidis/PYTHIA-dev/blob/main/docs/kdd-2026-presentation-material/PYTHIA-POSTER-KDD2026.pdf) presentation at [KDD '26](https://kdd2026.kdd.org/) !
- **[Aug 8, 2026]** Paper published and [available in the ACM Digital Library](https://dl.acm.org/doi/10.1145/3770855.3817968) !
- **[May 16, 2026]** *PYTHIA: Universal Question Answering over Knowledge Graphs* accepted as a full research paper at **KDD '26 (Research Track)** !

## Table of Contents
- [Repository structure](#repository-structure)
- [Quickstart](#quickstart)
  - [1. Environment](#1-environment)
  - [2. Indices](#2-indices)
    - [2a. Use pre-built indices](#2a-use-pre-built-indices)
    - [2b. Build your own indices](#2b-build-your-own-indices)
  - [3. Setup agent LLM](#3-setup-agent-llm)
  - [4. Setup graph search server, RDF endpoint and GoST endpoint](#4-setup-graph-search-server-rdf-endpoint-and-gost-endpoint)
  - [5. Use PYTHIA to generate answers for a dataset](#5-use-pythia-to-generate-answers-for-a-dataset)
  - [6. Evaluate outputs](#6-evaluate-outputs)
- [Reproducibility](#reproducibility)
  - [Knowledge Graph Versions Used](#knowledge-graph-versions-used)
  - [Systems](#systems)
  - [Metrics](#metrics)
- [Publication & Citation](#publication--citation)
- [Team & Authors](#team--authors)

## Repository structure
The repository is organized as follows:
- `_other_systems/` contains the versions of *Think-on-Graph (ToG)*, *Paths-over-Graph (PoG)*, *GRASP* and LLM baselines used in the evaluation with instructions on how to reproduce our results.
- `datasets/` contains the QA datasets used in the evaluation.
- `indices/` is the default location for placing system indices.
- `logs-and-results/` contains logs, raw results and metrics for the evaluations of *PYTHIA* and other systems.
- `src/` contains the source code of *PYTHIA*.
- `tools/` contains helper tools for creating indices, file manipulation and other auxiliary functions.
- `config.yaml` contains the configuration of *PYTHIA*.

*Additional information is provided in each sub-directory.*

## Quickstart

### 1. Environment

We recommend the use of *Conda* for dependency handling. The existence of a GPU is also recommended to speed-up embedding calculation.

To create the environment:
```sh
conda env create -f pythia.yml
```

To activate:
```sh
conda activate pythia
```

Additionally, you will need to install SimString in the same environment. Instructions can be found in the [official website](www.chokkan.org/software/simstring/).

### 2. Indices

*PYTHIA* utilizes indices for fast and accurate entity linking. You can either build your own or utilize our [pre-built indices](https://ai.di.uoa.gr/file-server/pythia/). Index folders must be placed in the `index_dir` specified in `config.yaml`

#### 2a. Use pre-built indices

If you choose to use pre-built indices, unzip their contents in `index_dir`. By default this is a directory named `indices/` at the project root.

Resulting structure:
```bash
indices/bestiary/
indices/dbpedia10/
indices/freebase/
```

#### 2b. Build your own indices

For knowledge graphs that don't have pre-built indices, or if you want more control over their contents, you can build your own indices. The following example is for building indices over Freebase.

**a)** Extract information about classes, entities and predicates from the knowledge graph source files. To do so, use `tools/elements_extractor`. Additional information can be found in the `tools/` directory.

```sh
java -jar target/elements-extractor-1.0-SNAPSHOT-jar-with-dependencies.jar \
  -i ~/freebase.nt \
  -o ~/extracts_freebase/ \
  -kg freebase \
  -l http://www.w3.org/2000/01/rdf-schema#label \
  -npl \
  -d http://rdf.freebase.com/ns/common.topic.description \
  -f -fc
```
##### *Note: The use of -Xmx -Xss flags might be required if the process runs out of memory for the given knowledge graph and arguments.*

This command takes an RDF dump of Freebase (`~/freebase.nt`) and generates a folder with TSV files (`~/extracts_freebase`). 

**b)** Generate class and entity indices. To do so, use `tools/all_indices/generator.py`.

```sh
python generator.py --input-dir ~/extracts_freebase --output-dir ~/indices_freebase
```

**c)** Move generated files to `index_dir`. Take the outputs of both commands (that is, the contents of `~/extracts_freebase/` and `~/indices_freebase/`) and put them in a new directory in the `index_dir` specified in `config.yaml`.

**d)** Compute class-predicate statistics. To do so, use `tools/class_predicates_extractor`.

```sh
java -jar target/class-predicates-extractor-1.0-SNAPSHOT-jar-with-dependencies.jar -i ~/freebase.nt -cf ~/extracts_freebase/classes.tsv -o {PATH_TO_FREEBASE_INDEX_DIR}
```
##### *Note: Make sure that after running this computation the output files are named `classes_predicates.tsv` and `classes_predicates_no_literals.tsv`.*

**e)** A knowledge graph structure index will be automatically created when you first use your index. This might take some time, but all subsequent uses will be much faster.

### 3. Setup agent LLM

To setup your LLM you need to provide an API key (and/or change model or provider). To use the default configuration simply set an OpenAI API key in `config.yaml` or in an environment variable named `OPENAI_API_KEY`.

### 4. Setup graph search server, RDF endpoint and GoST endpoint

**a)** *PYTHIA* requires a live RDF endpoint to execute queries. This endpoint should hold the same information as what was given for index creation. We recommend the use of *GraphDB* to simplify setup.

**b)** *PYTHIA* requires setting up a standalone graph search server. We split this from the body of *PYTHIA* for additional flexibility. Instructions for setting up the graph search server can be found in `tools/graph_minimizer/`. After setting up the graph search server its location must be given to *PYTHIA* via either setting it in `config.yaml` or setting the `GRAPH_SEARCH_SERVER` environment variable. 

##### Note: You can [download](https://ai.di.uoa.gr/file-server/pythia/graph-search/) pre-built search indices. To build the search indices see the relevant `README.md` file.

**c)** *PYTHIA* requires a *GoST* endpoint to function. *GoST* is a utility built using Apache Jena for query modification and formatting (you can see supported actions in the source code). Before running *PYTHIA* see how to run *GoST* (`tools/gost`).

### 5. Use *PYTHIA* to generate answers for a dataset

After having setup dependencies, indices and API key, you are ready to execute *PYTHIA*. In the following example we run *PYTHIA* over *WebQSP* from the project root directory (which is placed in our `HOME` directory on a Linux system).

```sh
python -m src.evaluation.qa_engine.pythia_generator \
  --index_name freebase \
  --dataset webqsp \
  --dataset_path ~/pythia/datasets/webqsp/data/WebQSP.test_clean.json \
  --bela --topic_entities_dataset ~/pythia/logs-and-results/nerd/baselines/bela_webqsp.json \
  --output_dir ~/pythia-results/ \
  --endpoint_server localhost:7200
```
##### Note: The use of *BELA* is optional, but is the default configuration. In this example we use pre-calculated *BELA* topic entities for package deployment simplicity. Alternatively, *BELA* can be ran as a standalone server (see `tools/bela_endpoint`).

### 6. Evaluate outputs

After *PYTHIA* finishes with the generation of the results we can run our evaluation scripts to calculate a number of metrics.

```sh
python -m src.evaluation.qa_engine.sparql_eval \
  --dataset webqsp \
  --dataset_path ~/pythia/datasets/webqsp/data/WebQSP.test_clean.json \
  --generated_file ~/pythia-results/PYTHIA_webqsp_results.json \
  --endpoint_server localhost:7200
```

In this example, results can be found in `~/pythia-results/`.

## Reproducibility

### Knowledge Graph Versions Used

An often overlooked aspect of reproducibility is the version of the underlying knowledge graph. To avoid this problem, we note the exact versions and sources of our knowledge graphs.

| Dataset | Knowledge Graph Version | Source |
|---|---|---|
| WebQSP | Latest Freebase dump | https://developers.google.com/freebase |
| CWQ | Latest Freebase dump | https://developers.google.com/freebase |
| QALD 9 | DBpedia 2016-10 | https://downloads.dbpedia.org/wiki-archive/dbpedia-version-2016-10.html |
| LC-QuAD | DBpedia 2016-04 | https://downloads.dbpedia.org/wiki-archive/dbpedia-version-2016-04.html |
| QALD 10 | QALD 10 Wikidata dump | https://zenodo.org/records/7496690 |
| BESTIARY | Knowledge graph from the official repository | https://github.com/danrd/sparqlgen |

For our experiments we use GraphDB 10.6.3.

### Systems

This repository contains the source code of *PYTHIA* and of prior works used in our experimental evaluation. We also provide pre-built knowledge graph indices, the exact versions of the datasets used and information about sourcing knowledge graph files. These should enable full reproducibility (with the caveat of randomness that is inherent in LLMs).

---

For running experiments with *PYTHIA* see this `README.md` file as well as the individual `README.md` files in the `tools/` directory. 

---

For running experiments with *ToG*, *PoG* and *GRASP* see `_other_systems`.

---

For running LLM baseline experiments see `src/baselines`.


### Metrics

In addition to our evaluated files, we include raw outputs for all systems evaluated, to allow future research efforts to re-evaluate outputs without needing to re-run systems.

## Publication & Citation

🔍 If you want to learn more about *PYTHIA* consider:
- Reading our [**KDD '26 paper**](https://dl.acm.org/doi/10.1145/3770855.3817968) !
- Taking a look at our [**poster**](https://github.com/SKefalidis/PYTHIA-dev/blob/main/docs/kdd-2026-presentation-material/PYTHIA-POSTER-KDD2026.pdf) and [**presentation slides**](https://github.com/SKefalidis/PYTHIA-dev/blob/main/docs/kdd-2026-presentation-material/PYTHIA-SLIDES-KDD-2026.pptx.pdf) !

📜 If you use *PYTHIA* in your research, please consider citing our work:

**Human Readable Format**
> Sergios-Anestis Kefalidis, Konstantinos Plas, and Manolis Koubarakis. 2026. PYTHIA: Universal Question Answering over Knowledge Graphs. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26). Association for Computing Machinery, New York, NY, USA, 2285–2296. https://doi.org/10.1145/3770855.3817968

**BibTex Format**
```bibtex
@inproceedings{kefalidis2026pythia,
  author    = {Kefalidis, Sergios-Anestis and Plas, Konstantinos and Koubarakis, Manolis},
  title     = {PYTHIA: Universal Question Answering over Knowledge Graphs},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  series    = {KDD '26},
  year      = {2026},
  pages     = {2285--2296},
  numpages  = {12},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  location  = {Republic of Korea},
  isbn      = {9798400722592},
  doi       = {10.1145/3770855.3817968},
  url       = {[https://doi.org/10.1145/3770855.3817968](https://doi.org/10.1145/3770855.3817968)},
  keywords  = {knowledge graph question answering, kg-rag, text-to-sparql}
}
```

## Team & Authors

<img align="right" src="https://github.com/AI-team-UoA/.github/blob/main/AI_LOGO.png?raw=true" alt="ai-team-uoa" width="200"/>

- [Sergios-Anestis Kefalidis](http://users.uoa.gr/~skefalidis/), Research Associate at the University of Athens, Greece
- [Kostas Plas](https://www.madgik.di.uoa.gr/el/people/msc-student/kplas), Research Associate at the University of Athens, Greece
- [Manolis Koubarakis](https://cgi.di.uoa.gr/~koubarak/), Professor at the University of Athens, Greece

This is a research project by the [AI-Team](https://ai.di.uoa.gr) of the Department of Informatics and Telecommunications at the University of Athens.
