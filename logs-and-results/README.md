## Logs and Results

This directory stores experimental outputs, evaluation files, and archived result bundles both for transparency as well as to enable future research efforts to calculate metrics without needing to re-run these systems on the same datasets.

Archives (`.tar.gz`) are used for files that are large.

### Structure

```text
logs-and-results/
├── ablation/         # ablation experiments
│   ├── beam-search/    # beam search ablations (many parameterized runs)
│   ├── no-bela/        # PYTHIA runs without BELA-based NERD
│   └── no-find/        # PYTHIA runs without entity linking tool
├── main/             # KGQA Systems (main tables)
│   ├── pythia/         # archived primary PYTHIA runs (.tar.gz)
│   ├── grasp/          # archived GRASP runs (.tar.gz)
│   ├── llm-baseline/   # baseline LLM outputs and evaluations
│   ├── llm-fewshot/    # few-shot LLM outputs (bela/oracle variants)
│   ├── llm-finetuning/ # fine-tuned LLM outputs and evaluations
│   ├── pog/            # PoG outputs and metrics
│   └── tog/            # ToG outputs (split by bela/oracle)
└── nerd/             # BELA results
    └── bela_*.json     # BELA entity linking artifacts per dataset
```

### Common File Types

- `*_results.json`, `*_generated.json`: raw generated predictions/results
- `*_results_full.json`, `*_generated_full.json`: extended result dumps
- `*_eval.json`: per-item evaluation output
- `*_metrics.json`, `*_eval_metrics.json`: aggregate metrics
- `*.jsonl`: line-delimited outputs (ToG)
- `*.tar.gz`: archived run bundles (*PYTHIA* and GRASP)

### Naming Conventions

- `bela` and `oracle` in filenames indicate the NERD/entity-linking setup.
- Dataset names are embedded in filenames (e.g., `cwq`, `webqsp`, `qald9`, `qald10`, `lc_quad`, `bestiary`).
- Ablation filenames may include parameter traces (for example in `ablation/beam-search/`).
