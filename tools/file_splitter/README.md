# file_splitter

Small shell helper for splitting very large files into multiple chunks. Used to split up large RDF dumps to facilitate multi-threaded work in tools that support it (e.g., `elements_extractor`).

## Usage Example

```bash
bash split.sh ~/freebase.nt ~/freebase_chunks 20
```