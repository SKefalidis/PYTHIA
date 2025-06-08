# bela_endpoint

Runs a lightweight *BELA* API service. This can be used either in real-time with *PYTHIA* or as a standalone for evaluation or offline entity linking.

## Setup BELA

To setup *BELA* please see the [official repo](https://github.com/facebookresearch/BELA).

## Contents

- `bela_endpoint.py`: FastAPI server with `/nerd` endpoint.
- `bela_test.py`: simple request smoke test.

## Endpoint

- `POST /nerd`
  - Request body: `{ "question": "..." }`
  - Response body: `{ "result": [...] }`

## Usage Example

```bash
python bela_endpoint.py --host 0.0.0.0 --port 8001
```

Then test:

```bash
python bela_test.py
```

## Arguments

- `--host`: bind host (default: `0.0.0.0`).
- `--port`: bind port (default: `8001`).

## Notes

- The service initializes `BELA(device="cuda:0")`; GPU availability is expected.
- CORS is open (`*`) for convenience.
