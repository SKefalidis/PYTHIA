# Other Systems

This direcotry contains the exact versions of *ToG*, *PoG* and *GRASP* which were used to re-evaluate those systems. We have made slight modifications to *ToG* and *PoG* to transfer them to different knoweldge graphs, without negatively impacting performance. Similarly, we have made an even smaller modification in *GRASP* to make evaluation easier for our pipeline.

All systems are documented by the original authors, and our evaluator scripts are self-documented (`--help`).

## Contents

- `grasp/`: Contains source files for *GRASP*. The system is meant to be setup as detailed by the original authors. Afterwards, `pythia/src/evaluation/qa_engine/grasp_generator.py` (not in this folder) is used to generate results for each dataset. More detailed instructions and re-production scripts are currently being written. [Official code repository](https://github.com/ad-freiburg/grasp/tree/main).
- `PoG/`: Contains source files for *Paths-over-Graph (PoG)* ([official code repository](https://github.com/SteveTANTAN/PoG/tree/main)). The modifications made by us to support additional knowledge graphs and make *Pog* easier to run with time and memory limits are also available as a `.patch` file.
- `ToG/`: Contains source files for *Think-on-Graph (ToG)* ([official code repository](https://github.com/DataArcTech/ToG)). The modifications made by us to support additional knowledge graphs are also available as a `.patch` file.

After experimental results are produced, we the same evaluation files that *PYTHIA* uses to produces the final metrics.

## How to run experiments

### GRASP

*GRASP* is the only system under active development among prior systems that were used in the evaluation of *PYTHIA*. For that reason it is a bit trickier to set-up using the latest development branch.

We provide the version that we used, which has been modified to return additional information when answering requests, to facilitate our analysis.

To run it:

1. Set up the system as detailed by the authors

2. Start a *GRASP* server using the configuration file in `grasp/configs/serve.yaml` (make sure to add pointers to your RDF endpoint(s)).

3. Run the generation script that exists in the source folder of *PYTHIA* (because it utilizes some *PYTHIA* libraries) `pythia/src/evaluation/qa_engine/grasp_generator.py`.

Example without oracle entities:

```
python -m src.evaluation.qa_engine.grasp_eval \
    --dataset beastiary \
    --dataset_path ~/pythia/datasets/beastiary/beastiary_with_qald_format.json \
    --output_dir ~/grasp-results/zero-shot \
    --kg bestiary
```

Example with oracle entities:

```
python -m src.evaluation.qa_engine.grasp_eval \
    --dataset beastiary \
    --dataset_path ~/pythia/datasets/beastiary/beastiary_with_qald_format.json \
    --oracle \
    --topic_entities_dataset ~/pythia/datasets/beastiary/bestiary_topic_entities.json \
    --output_dir ~/grasp-results/zero-shot-oracle \
    --kg bestiary
```

### Paths over Graph

1. Set up the system as detailed by the original authors.

2. Instead of the original way of execution we use the script `PoG/pog_generator.py`. This controls the entire process, enforcing time limits and memory usage requirements.

Example of running PoG for BESTIARY, using a three minute timeout (default):

```
python pog_generator.py \
    --dataset bestiary \
    --output_dir ~/pog-results \ 
    --endpoint_url http://localhost:7200/repositories/beastiary \
    --user user \
    --password password
```


### Think on Graph

1. Set up a Python environment as detailed by the original authors. (*Note:* our modified version utilizes an RDF endpoint for all knowledge graphs, simplifying setup).

2. Run following the instructions of the original authors. (*Note:* for any target KG use `ToG/ToG/main_freebase.py`).

## How to evaluate outputs

### SPARQL Responses

For *GRASP* use the same evaluation script that is used for *PYTHIA* (as detailed in the main `README.md` file).

Example for evaluating the outputs of *GRASP* over WebQuestions Semantic Parsing (WebQSP):

```
python -m src.evaluation.qa_engine.sparql_eval \
    --dataset webqsp \
    --dataset_path ~/pythia/datasets/webqsp/data/WebQSP.test_clean.json \
    --generated_file ~/grasp-results/zero-shot/GRASP_webqsp_test.json \
    --endpoint_graphdb_server localhost:7200
```

### Textual Responses

For *ToG* and *PoG* you must first extract answers from their natural language outputs. To do so, use `tools/extract_nl_answers`.

Example:

```
python -m tools.extract_nl_answers.extract_nl_answers \
    --file ~/pog-results/PoG_lc-quad-1_results.json \
    --question_key question \
    --answer_key results
```

Then use the generated file for evaluation with `pythia/src/evaluation/qa_engine/text_eval.py`.

Example:

```
python -m src.evaluation.qa_engine.text_eval \
    --generated_file ~/pog-results/PoG_lc-quad-1_results_csv_extracted.json \
    --dataset lc-quad-1 \
    --dataset_path ~/pythia/datasets/lc_quad_1/lc_quad_1_test-data.json \
    --endpoint_server localhost:7200 \
    --system pog
```