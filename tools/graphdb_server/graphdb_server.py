from __future__ import annotations

import atexit
import os
import signal
import subprocess
import threading
import argparse
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, jsonify, request


SERVER_PORT = 65201
GRAPHDB_PORT = 65200

USERNAME = None
PASSWORD = None

GRAPHDB_DIR = None
GRAPHDB_BIN = "./graphdb-10.6.3/bin/graphdb"
GRAPHDB_ARGS = ["-Dgraphdb.engine.onheap.allocation=false"]

_lock = threading.Lock()
_graphdb_proc: Optional[subprocess.Popen] = None


def _is_running() -> bool:
    return _graphdb_proc is not None and _graphdb_proc.poll() is None


def start_graphdb() -> Dict[str, str]:
    global _graphdb_proc
    with _lock:
        if _is_running():
            return {"status": "already-running", "pid": str(_graphdb_proc.pid)}

        # Start GraphDB in its own process group so we can terminate cleanly later.
        _graphdb_proc = subprocess.Popen(
            [os.path.join(GRAPHDB_DIR, GRAPHDB_BIN)] + GRAPHDB_ARGS,
            preexec_fn=os.setsid,
        )
        return {"status": "started", "pid": str(_graphdb_proc.pid)}


def stop_graphdb() -> Dict[str, str]:
    global _graphdb_proc
    with _lock:
        if not _is_running():
            _graphdb_proc = None
            return {"status": "not-running"}

        try:
            os.killpg(_graphdb_proc.pid, signal.SIGTERM)
            _graphdb_proc.wait(timeout=10)
            status = "stopped"
        except subprocess.TimeoutExpired:
            os.killpg(_graphdb_proc.pid, signal.SIGKILL)
            status = "killed"
        finally:
            _graphdb_proc = None
        return {"status": status}


def restart_graphdb() -> Dict[str, str]:
    was_running = _is_running()
    stop_info = stop_graphdb()
    start_info = start_graphdb()
    return {
        "was_running": str(was_running).lower(),
        "stop": stop_info,
        "start": start_info,
    }


def load_repository(endpoint: str) -> Dict[str, str]:
    print(f"Loading repository for endpoint: {endpoint}")
    
    cmd = [
        "curl",
        "-u",
        f"{USERNAME}:{PASSWORD}",
        "-X",
        "POST",
        endpoint,
    ]
    
    print(f"Executing command: {' '.join(cmd)}")

    completed = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "returncode": str(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


app = Flask(__name__)


@app.route("/clear", methods=["POST"])
def clear() -> object:
    result = restart_graphdb()
    return jsonify(result)


@app.route("/load", methods=["POST"])
def load() -> object:
    data = request.get_json(force=True)
    endpoint = data.get("endpoint")
    
    if not endpoint:
        return jsonify({"error": "Missing 'endpoint' in request body"}), 400

    result = load_repository(endpoint)
    return jsonify(result)


@app.route("/")
def root() -> object:
    return jsonify({"status": "ok", "graphdb_running": _is_running()})


def _shutdown() -> None:
    stop_graphdb()


def main() -> None:
    atexit.register(_shutdown)
    print(f"GraphDB helper listening on http://0.0.0.0:{SERVER_PORT}")
    app.run(host="0.0.0.0", port=SERVER_PORT, threaded=True)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphDB server helper")
    
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Username for GraphDB authentication",
    )
    
    parser.add_argument(
        "--password",
        type=str,
        required=True,
        help="Password for GraphDB authentication",
    )
    
    parser.add_argument(
        "--graphdb-dir",
        type=Path,
        required=True,
        help="Path to the GraphDB installation directory",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help=f"Port for the graphdb server to listen on (default: {GRAPHDB_PORT})",
    )
    
    args = parser.parse_args()

    GRAPHDB_DIR = args.graphdb_dir
    USERNAME = args.username
    PASSWORD = args.password
    GRAPHDB_PORT = args.port

    main()