# extract_nl_answers

Post-processing helper that converts natural-language answers into structured CSV-style answers using an LLM. 

This is useful because *ToG* and *PoG* outputs (used in our evaluation) might be formatted incorrectly which lowers accuracy. This way we run a focused LLM pass for proper formatting.

## Usage Example

```bash
python extract_nl_answers.py \
  --file ~/pythia-results/PYTHIA_webqsp_results.json \
  --question_key question \
  --answer_key answer
```

This uses [litellm](https://github.com/BerriAI/litellm), so it requirs that an appropriate model is setup (we use GPT-4.1-mini by default, so the user needs to set the enviornment variable `OPENAI_API_KEY`).

## Arguments

- `--file` (required): input JSON with question/answer entries.
- `--question_key`: key name for the question field.
- `--answer_key`: key name for the answer field.

## Output

- Creates a new file next to the input:
  - `<input_name>_csv_extracted.json`
- Adds a `csv_answer` field per output item.
