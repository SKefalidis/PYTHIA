# graphdb_server

Flask helper service to start/stop GraphDB and trigger repository loading for *PYTHIA* setup workflows. This was used by us to make our experimental evaluation more consistent (always start with a fresh GraphDB instance).

**This is not required for running *PYTHIA*.**

## Script

- `graphdb_server.py`

## Usage

```bash
python graphdb_server.py \
  --username <graphdb_user> \
  --password <graphdb_password> \
  --graphdb-dir ~/pythia
```

## Arguments

- `--username` (required): GraphDB username.
- `--password` (required): GraphDB password.
- `--graphdb-dir` (required): directory that contains the configured GraphDB binary path used by the script.
- `--port`: GraphDB port override (optional).

## Service endpoints

- `GET /`: health + running status.
- `POST /clear`: restart GraphDB process.
- `POST /load`: load repository via `curl` to provided endpoint.

Example `POST /load` body:

```json
{ "endpoint": "http://localhost:7200/rest/repositories" }
```
