# PYTHIA — Agent Development & Contribution Guide

This document provides autonomous coding agents (e.g., Google Jules, Antigravity) and human contributors with essential context, architectural layout, commands, and invariants to navigate, modify, and test PYTHIA effectively.

---

## 1. Project Overview & Quick Context

* **Domain:** Zero-shot, Knowledge-Graph-Agnostic Question Answering (**KGQA**).
* **Primary Language:** Python 3.10+ (core logic) + Java 17 / Maven (specialized indexing tools under `tools/`).
* **Core Philosophy:** Plug-and-play KGQA over arbitrary Knowledge Graphs (Wikidata, DBpedia, Freebase, custom KGs) without training or fine-tuning.
* **Primary Entry Points:**
  * **Interactive CLI:** `python -m src.app.cli`
  * **API Server:** `python -m src.app.server`
  * **Evaluation Harness:** Evaluation-related shared code is in `src/evaluation/`, with specific evaluation scripts organized in:
    * `src/evaluation/nerd/`: Named Entity Recognition & Disambiguation evaluations (e.g., `nerd_evaluation.py`, `dbpedia_eval.py`).
    * `src/evaluation/qa_engine/`: QA engine benchmarks, SPARQL, and graph search evaluations (e.g., `pythia_generator.py`, `sparql_eval.py`, `graph_search_eval.py`).

---

## 2. Repository Layout & Architecture

```text
├── configs/               # YAML configuration files (core.yml, endpoints.yml, graphs.yml, app.yml)
├── datasets/              # QA benchmark datasets (WebQSP, CWQ, GrailQA, QALD, etc.)
├── indices/               # Default directory for entity linking & graph search indices
├── src/                   # Main Python application package
│   ├── app/               # CLI (`cli.py`), API server (`server.py`), and runner (`app.py`)
│   ├── config.py          # Pydantic configuration schemas and YAML loader
│   ├── core/              # Core agent reasoning engine
│   │   ├── agent/         # Main loop (`pythia_agent.py`) and prompt templates (`pythia_agent_prompts.py`)
│   │   └── tools/         # Agent tools:
│   │       ├── direct_retrieval/    # SimString, BM25, and FAISS entity linking
│   │       ├── graph_exploration/   # Graph search client & stepwise search over KGs
│   │       └── query_execution/     # SPARQL generation, validation, and execution
│   ├── dataloaders/       # Dataset loaders for standard benchmarks
│   ├── evaluation/        # Evaluation harness & metrics
│   │   ├── nerd/          # Entity linking / NERD evaluation scripts
│   │   └── qa_engine/     # End-to-end QA generation & evaluation scripts
│   ├── knowledge_graphs/  # KG abstractions and adapters (Wikidata, DBpedia, Freebase, CustomKG)
│   ├── query_index/       # Query database for few-shot in-context learning
│   └── utils.py           # Core utilities (SPARQL execution, URI prefix compression, LLM wrappers)
├── tools/                 # Auxiliary indexing and preprocessing tools (Java + Python)
│   ├── elements_extractor/          # Java/Maven tool for parsing N-Triples and extracting classes/predicates
│   ├── class_predicates_extractor/  # Java/Maven tool for hierarchy and predicate mapping
│   ├── faiss_index_generator/       # Python FAISS dense index builder
│   └── simstring_index_generator/   # Python SimString fuzzy index builder
└── pythia-cpu.yml         # Conda environment specification
```

### Subsystem Invariants
1. **Configuration (`src/config.py`):**
   * All configuration classes inherit from Pydantic `BaseModel`.
   * YAML files in `configs/` support dynamic environment variable overrides with the syntax `env(VAR_NAME:default_value)`.
   * **Rule:** Never hardcode endpoint URLs or API credentials in source files.
2. **Prompts (`src/core/agent/pythia_agent_prompts.py`):**
   * Keep prompt definitions centralized.
   * Ablation flags in `configs/core.yml` control which prompt components are enabled.

---

## 3. Environment & Setup Commands

### Environment Activation
The project is built around **Python 3.10**:
```bash
# Conda activation
conda activate pythia

# Alternatively with uv / venv (if configured)
source .venv/bin/activate
```

### Installation
```bash
# Using Conda
conda env create -f pythia-cpu.yml

# Fast pip / uv install
pip install -r requirements.txt  # or: uv pip install -e .
```

---

## 4. Headless Testing & Verification (Critical for Jules & Agents)

Autonomous agents operate in headless cloud sandboxes without live network access to production endpoints or paid API keys.

### 4.1 Running Tests
```bash
# Run the full test suite
pytest

# Run a specific unit test
pytest tests/unit/test_utils.py -v
```

### 4.2 Mocking Invariants
* **LLM Calls:** Any test touching agent reasoning must mock `llm_call` in `src/utils.py`. Do not invoke live OpenAI / Gemini endpoints in automated tests.
* **SPARQL Endpoints:** Mock `return_sparql_query_results` in `src/utils.py` with predefined result tuples. Do not attempt to connect to live GraphDB/Virtuoso instances during unit tests.
* **GoST Server (SPARQL Validation):** GoST is a standalone service used strictly for **SPARQL validation** (it does not perform graph search). Mock its validation responses during unit tests.
* **Graph Search Server (Exploration):** Graph search is a separate service handled by `src/core/tools/graph_exploration/graph_search_client.py`. Mock its search responses during unit tests.

### 4.3 Code Formatting
* Ensure new or modified code strictly matches the formatting and coding style of the existing code around it.

---

## 5. Contribution & Development Guidelines

### 5.1 Mandatory AI Review Notice in Conversation
> [!IMPORTANT]
> **DO NOT add disclaimer comments or attribution headers directly into source code files.**
> 
> Instead, whenever files are created or substantially modified by an AI agent, the agent must provide an explicit notice in the conversation (and/or pull request description) identifying all modified files and requesting human review before merging into main or deploying to production.

### 5.2 Mandatory AI Assistance Credit in Commits
> [!IMPORTANT]
> **All commits and code changes that involve significant AI contributions must credit the AI agent in the commit message.**
>
> Include the `Assisted-by:` trailer (without an email address) in the commit message body:
> ```text
> Assisted-by: Antigravity
> ```
> *(Adjust the agent name for other autonomous agents, e.g. `Assisted-by: Google Jules`)*

### 5.3 Scoping & Commits
1. **Atomic Scope (Single PR Rule):**
   * Keep PRs strictly bounded to one feature, bug fix, or tool.
   * Do not mix refactoring with behavioral changes.
2. **Commit Convention:**
   * Use Conventional Commits: `fix: ...`, `feat: ...`, `refactor: ...`, `test: ...`, `docs: ...`.
   * Include `Assisted-by:` metadata (without an email address) whenever AI assistance has played a significant role in authoring the changes.
3. **No Unconstrained Searches:**
   * Keep tool calls and file scans targeted to relevant directories (`src/`, `tools/`). Avoid broad scans across parent directories.
