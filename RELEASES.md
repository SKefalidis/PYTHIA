# Release Notes

This document highlights new features, major changes, minor improvements and bug-fixes for each release of PYTHIA. 

---

## [v1.0.1-dev1] - 2026-10-30

This is a development release containing documentation updates, general housekeeping, bug-fixes and a few minor features. 

PYTHIA is currently under active development, however this version should maintain performance parity with the published KDD 2026 results, while being easier to deploy and use.

**Main Features**
- Updated README.md file with clearer instructions, updated links and citation information.
- Improved deployment, error reporting, configuration and other user experience enhancements.
- Updated auxiliary code to use the latest versions of the core components (auxiliary components would not work due to being out of sync).
- Fixed a bug where IRI compression would not be applied in various cases. This is likely to lead to considerable improvements in the token efficiency of PYTHIA, and might also lead to improved performance.
- **Removed code unrelated to PYTHIA**. For any auxiliary files (e.g., experimental baselines, please release `v1.0.0-kdd-2026`)

**Other Changes**
- Added support for n-quad files in KG indexing.
- Removed unused files.
- Refactored parts of the codebase and fixed.
- Add support for TEXT2SPARQL challenge datasets.
- Added KDD 2026 presentation slides and poster.

## [v1.0.0-kdd-2026] - 2026-06-01

PYTHIA is a transferable KGQA engine with strong generalizability. As such it is highly flexible and can be used as a drop-in solution over any RDF knowledge graph.

This is the initial release of PYTHIA's codebase as it was developed and used in the KDD '26 paper "PYTHIA: Universal Question Answering over Knowledge Graphs".

Although the labeling is v1.0.0, this is not meant for production, instead it serves as the baseline implementation of the proposed system and is highly likely to include bugs and compatibility issues.

**Initial code release.** This version of the codebase was used for the original KDD 2026 submission and refined during the rebuttal phase. It includes the exact versions of the baselines that PYTHIA was compared to for improved reproducibility.