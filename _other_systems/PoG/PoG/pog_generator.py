from ast import alias
import os
import sys
import json
import argparse
import tempfile
import requests
import subprocess
import time
import threading
import globals
import signal
import psutil

from typing import Any, Dict, List
from tqdm import tqdm
from utils import *

# ------------------------
# ----- File Writing -----
# ------------------------

def atomic_write(filepath: str, data: Any, mode: str = "w") -> None:
    """Atomically write JSON data to filepath."""
    dir_name = os.path.dirname(filepath)
    os.makedirs(dir_name, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode=mode, dir=dir_name, delete=False) as tmp_file:
        json.dump(data, tmp_file, indent=4)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_name = tmp_file.name
    os.replace(temp_name, filepath)
    
def persist_all(
    base_path: str,
    results: List[Dict[str, Any]],
    results_full: List[Dict[str, Any]],
    metrics_summary: Dict[str, Any],
) -> None:
    save_targets = [
        (results, base_path + "_results.json"),
        (results_full, base_path + "_results_full.json"),
        (metrics_summary, base_path + "_metrics.json"),
    ]
    for data, path in save_targets:
        atomic_write(path, data)
        
# -------------------
# ----- Metrics -----
# -------------------
    
def extract_metrics(resp: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(resp, dict):
        return {}
    if isinstance(resp.get("metrics"), dict):
        return resp["metrics"]
    output = resp.get("output")
    if isinstance(output, dict) and isinstance(output.get("metrics"), dict):
        return output["metrics"]
    return {}
    
def merge_metrics(total: Dict[str, float], metrics: Dict[str, Any]) -> None:
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            total[key] = total.get(key, 0.0) + float(val)
            
# -------------------
# ----- PoG API -----
# -------------------

PORT_OFFSET: int = 0

proc_single: subprocess.Popen = None
SINGLE_ENDPOINT = "http://localhost:" + str(12345 + PORT_OFFSET)

proc_multi: subprocess.Popen = None
MULTI_ENDPOINT = "http://localhost:" + str(12346 + PORT_OFFSET)

SCRIPT_DIR = './'
DATASET_NAME = None
ENDPOINT = None

POG_CONDA_PYTHON = "/home/conda/miniconda3/envs/pog/bin/python"

# Memory limit in MB for PoG servers
MEMORY_LIMIT_MB = 50000.0 # 50 GB

def kill_server_safely(proc):
    """
    Kills a process AND all its children/grandchildren recursively using psutil.
    """
    if proc is None:
        return

    pid = proc.pid
    print(f"Attempting to kill process tree for PID {pid}...")

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"PID {pid} already dead.")
        return

    # 1. Gather all children (recursive=True gets grandchildren too)
    try:
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []

    # 2. Kill children first
    for child in children:
        try:
            print(f" - Killing child PID {child.pid}")
            child.kill()  # Sends SIGKILL immediately
        except psutil.NoSuchProcess:
            pass

    # 3. Kill the parent
    try:
        print(f" - Killing parent PID {parent.pid}")
        parent.kill()
        parent.wait(5) # clean up zombie state
    except psutil.NoSuchProcess:
        pass
    except subprocess.TimeoutExpired:
        print(f"Parent PID {pid} refused to die (zombie).")

    print(f"Process tree {pid} terminated.")

def start_pog_server() -> None:
    global proc_single, proc_multi
    
    # ----- PoG Single -----
    if proc_single:
        print("Restarting PoG single server...")
        proc_single = kill_server_safely(proc_single)
    else:
        print("Starting PoG single server...")
    cmd = [POG_CONDA_PYTHON, "PoG_single_web.py", DATASET_NAME, "sum", "13", "PoG", "gpt4", "3", ENDPOINT, globals.get_user(), globals.get_password(), str(PORT_OFFSET)]
    print(f"Executing command: {' '.join(cmd)}")
    
    with open(f"pog_single_{DATASET_NAME}_stdout.log", "a") as stdout_file, \
     open(f"pog_single_{DATASET_NAME}_stderr.log", "a") as stderr_file:
        proc_single = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            # stdout=stdout_file,
            # stderr=stderr_file,
            start_new_session=True
        )
    
    time.sleep(5)
    if check_pog_server(SINGLE_ENDPOINT):
        print(f"Started PoG single server with PID: {proc_single.pid}")
    else:
        print("Failed to start PoG single server.")
        
    # ----- PoG Multi -----
    if proc_multi:
        print("Restarting PoG multi server...")
        proc_multi = kill_server_safely(proc_multi)
    else:
        print("Starting PoG multi server...")
    cmd = [POG_CONDA_PYTHON, "PoG_multi_web.py", DATASET_NAME, "sum", "13", "PoG", "gpt4", "3", ENDPOINT, globals.get_user(), globals.get_password(), str(PORT_OFFSET)]
    print(f"Executing command: {' '.join(cmd)}")
    
    with open(f"pog_multi_{DATASET_NAME}_stdout.log", "a") as stdout_file, \
     open(f"pog_multi_{DATASET_NAME}_stderr.log", "a") as stderr_file:
        proc_multi = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            # stdout=stdout_file,
            # stderr=stderr_file,
            start_new_session=True
        )
    
    time.sleep(5)
    if check_pog_server(MULTI_ENDPOINT):
        print(f"Started PoG multi server with PID: {proc_multi.pid}")
    else:
        print("Failed to start PoG multi server.")
    
def check_pog_server(server_url: str, timeout: int = 3) -> bool:
    health_url = f"{server_url}/health"
    try:
        response = requests.get(health_url, timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False
    
def get_current_metrics_from_server(server_url):
    print(f"Fetching metrics from {server_url}...")
    try:
        response = requests.get(f"{server_url}/metrics", timeout=3)
        if response.status_code == 200:
            print(response.json())
            return response.json().get("metrics", {})
        else:
            print(f"Failed to fetch metrics, status code: {response.status_code}")
            return {}
    except requests.RequestException as e:
        print(f"Error fetching metrics from {server_url}: {e}")
        return {}
    
def check_pog_server_memory_usage() -> float:
    memory_single_url = f"{SINGLE_ENDPOINT}/memory"
    memory_multi_url = f"{MULTI_ENDPOINT}/memory"
    total_memory = 0.0
    try:
        response_single = requests.get(memory_single_url, timeout=3)
        if response_single.status_code == 200:
            total_memory += float(response_single.json().get("memory_usage_mb", 0.0))
    except requests.RequestException:
        pass
    try:
        response_multi = requests.get(memory_multi_url, timeout=3)
        if response_multi.status_code == 200:
            total_memory += float(response_multi.json().get("memory_usage_mb", 0.0))
    except requests.RequestException:
        pass
    return total_memory

def call_pog_api(
    question: str,
    topic_entities: Dict[str, str],
    question_id: str,
    timeout: int = 180,
) -> Dict[str, Any]:
    global proc_single, proc_multi
    
    # PoG Setup
    if len(topic_entities) <= 1:
        server_url = SINGLE_ENDPOINT
    else:
        server_url = MULTI_ENDPOINT
    
    def ensure_pog_server_ready() -> None:
        global proc_single, proc_multi
        
        while not check_pog_server(server_url):
            wait_time = 5
            if proc_single:
                proc_single = kill_server_safely(proc_single)
            if proc_multi:
                proc_multi = kill_server_safely(proc_multi)
            time.sleep(wait_time)
            if check_pog_server(SINGLE_ENDPOINT) and not proc_single:
                input("PoG single server is running but process handle is missing. Please restart the script after killing the PoG single server. Press any key to continue...")
            if check_pog_server(MULTI_ENDPOINT) and not proc_multi:
                input("PoG multi server is running but process handle is missing. Please restart the script after killing the PoG multi server. Press any key to continue...")
            start_pog_server()
            # time.sleep(wait_time)  # Wait for servers to start
            wait_time += 5
            
    ensure_pog_server_ready()
    
    # Memory Monitoring
    stop_monitoring = threading.Event()
    memory_limit_hit = False
    current_metrics = {}

    def monitor():
        nonlocal memory_limit_hit
        nonlocal current_metrics
        while not stop_monitoring.is_set():
            try:
                current_metrics_new = get_current_metrics_from_server(server_url)
                if current_metrics and current_metrics_new:
                    current_metrics = current_metrics_new
                elif current_metrics and not current_metrics_new:
                    pass
                elif not current_metrics and current_metrics_new:
                    current_metrics = current_metrics_new
                elif not current_metrics and not current_metrics_new:
                    current_metrics = {}
                current_mem = check_pog_server_memory_usage()
                if current_mem > MEMORY_LIMIT_MB:
                    print(f"Memory limit exceeded: {current_mem} MB > {MEMORY_LIMIT_MB} MB")
                    print("Terminating PoG servers...")
                    memory_limit_hit = True
                    # Forcefully kill servers to abort the blocking request in main thread
                    if proc_single: proc_single = kill_server_safely(proc_single)
                    else:
                        print("Please manually kill the PoG single server.")
                        input("Press Enter to continue after restarting the PoG single server...")
                    if proc_multi: proc_multi = kill_server_safely(proc_multi)
                    else:
                        print("Please manually kill the PoG multi server.")
                        input("Press Enter to continue after restarting the PoG multi server...")
                    ensure_pog_server_ready()
                    break
            except Exception:
                # Ignore monitoring errors (e.g. server down) to avoid crashing main thread
                pass
            time.sleep(10) # Check every 10 seconds

    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()
    
    # PoG Execution
    payload = {
        "question": question,
        "topic_entities": topic_entities,
        "question_id": question_id
    }
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{server_url}/query",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        elapsed = time.time() - start_time
        ensure_pog_server_ready()
        
        if memory_limit_hit:
            error_msg = "PoG server memory limit exceeded"
        else:
            error_msg = str(e.response.text) if e.response is not None else str(e)
        return {
            "question":question, 
            "error": error_msg,
            "results": "ERROR", 
            "reasoning_chains": [], 
            "answer_type": "ERROR",
            "metrics" : {
                "TIME": elapsed,
                "SPARQL_CALLS": current_metrics.get("SPARQL_CALLS", "Unknown"),
                "SPARQL_TIME": current_metrics.get("SPARQL_TIME", "Unknown"),
                "LLM_CALLS": current_metrics.get("LLM_CALLS", "Unknown"),
                "LLM_INPUTS": current_metrics.get("LLM_INPUTS", "Unknown"),
                "LLM_OUTPUTS": current_metrics.get("LLM_OUTPUTS", "Unknown"),
                "LLM_TIME": current_metrics.get("LLM_TIME", "Unknown")
            }
        }
    finally:
        # Stop the monitor thread regardless of success or failure
        stop_monitoring.set()
        monitor_thread.join()

    json_result = response.json()['result']
    if json_result is None or json_result == {}:
        return {
            "question":question, 
            "error": "Empty response from PoG server",
            "results": "ERROR_EMPTY", 
            "reasoning_chains": [], 
            "answer_type": "ERROR_EMPTY",
            "metrics" : {
                "TIME": time.time() - start_time,
                "SPARQL_CALLS": current_metrics.get("SPARQL_CALLS", "Unknown"),
                "SPARQL_TIME": current_metrics.get("SPARQL_TIME", "Unknown"),
                "LLM_CALLS": current_metrics.get("LLM_CALLS", "Unknown"),
                "LLM_INPUTS": current_metrics.get("LLM_INPUTS", "Unknown"),
                "LLM_OUTPUTS": current_metrics.get("LLM_OUTPUTS", "Unknown"),
                "LLM_TIME": current_metrics.get("LLM_TIME", "Unknown")
            }
        }
    else:
        return response.json()['result']

# ----------------
# ----- Main -----
# ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GRASP text-to-SPARQL over a dataset and save outputs.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (for DatasetFactory)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write outputs")
    parser.add_argument("--endpoint_url", type=str, required=True, help="SPARQL endpoint URL for executing generated queries")
    parser.add_argument("--timeout", type=int, default=180, help="Request timeout in seconds")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--dataset_name", type=str, default=None, help="Optional dataset name override for filenames")
    parser.add_argument("--user", type=str, default="user")
    parser.add_argument("--password", type=str, default="user")
    parser.add_argument("--port_offset", type=int, default=0, help="Port offset for PoG servers (optional, default is 0)")
    args = parser.parse_args()
    
    globals.set_user(args.user)
    globals.set_password(args.password)
    
    global PORT_OFFSET, SINGLE_ENDPOINT, MULTI_ENDPOINT
    PORT_OFFSET = args.port_offset
    SINGLE_ENDPOINT = "http://localhost:" + str(12345 + PORT_OFFSET)
    MULTI_ENDPOINT = "http://localhost:" + str(12346 + PORT_OFFSET)
    
    global DATASET_NAME
    DATASET_NAME = args.dataset
    global ENDPOINT
    ENDPOINT = args.endpoint_url
    
    global proc_single, proc_multi

    dataset, question_string = prepare_dataset(args.dataset)
    dataset_name = args.dataset_name or args.dataset
    base_path = os.path.join(args.output_dir, f"PoG_{dataset_name}")

    start_idx = 0

    if not args.overwrite and os.path.exists(base_path + "_results.json"):
        print(f"Outputs already exist at {base_path}_*.json. If you want to continue input 'y', otherwise rerun using --overwrite to regenerate")
        decision = input().strip().lower()
        if decision != 'y':
            sys.exit(0)
        if os.path.exists(base_path + "_results.json"):
            with open(base_path + "_results.json", "r") as infile:
                existing_results = json.load(infile)
                start_idx_results = len(existing_results)
        if os.path.exists(base_path + "_results_full.json"):
            with open(base_path + "_results_full.json", "r") as infile:
                existing_results_full = json.load(infile)
                start_idx_results_full = len(existing_results_full)
        if os.path.exists(base_path + "_metrics.json"):
            with open(base_path + "_metrics.json", "r") as infile:
                existing_metrics = json.load(infile)
                # start_idx_metrics = existing_metrics.get("QUESTIONS", 0)
        if start_idx_results != start_idx_results_full:
            print("Warning: Inconsistent number of existing entries in results files.")
            print(f"Results: {start_idx_results}, Results Full: {start_idx_results_full}")
            sys.exit(1)
        else:
            start_idx = start_idx_results

    if start_idx == 0:
        results: List[Dict[str, Any]] = []
        results_full: List[Dict[str, Any]] = []
        aggregate_metrics: Dict[str, float] = {}
        success_count = 0
        total_elapsed = 0.0
        
        total_questions = len(dataset)
        total_elapsed_with_errors = 0.0
        print("Starting evaluation from scratch...")
    else:
        with open(base_path + "_results.json", "r") as infile:
            results = json.load(infile)
        with open(base_path + "_results_full.json", "r") as infile:
            results_full = json.load(infile)
        with open(base_path + "_metrics.json", "r") as infile:
            existing_metrics = json.load(infile)
            aggregate_metrics = {k: float(v) for k, v in existing_metrics.items() if k not in ["QUESTIONS", "TIME", "TIME_PER_QUESTION", "TOTAL_QUESTIONS", "TOTAL_TIME_WITH_ERRORS", "TIME_PER_QUESTION_WITH_ERRORS"]}
            success_count = existing_metrics.get("QUESTIONS", 0)
            total_elapsed = existing_metrics.get("TIME", 0.0)
            total_questions = len(dataset)
            total_elapsed_with_errors = existing_metrics.get("TOTAL_TIME_WITH_ERRORS", 0.0)
        print(f"Resuming from index {start_idx}...")

    for idx in tqdm(range(start_idx, len(dataset))):
        entry = dataset[idx]
        question = entry[question_string]
        
        if args.dataset.lower() in ['webqsp', 'cwq']:
            topic_entities = {"ns:"+k: v for k, v in entry['topic_entity'].items()}  
        elif args.dataset.lower() in ['qald-10', 'qald-9', 'lc-quad-1', 'lc-quad-2', 'spinach', 'bestiary', 'geoq1089']:
            topic_entities = {k: v for k, v in entry['topic_entities'].items()}
        else:
            raise ValueError(f"Dataset {args.dataset} not supported for PoG evaluation.")
        
        if check_pog_server(SINGLE_ENDPOINT) and not proc_single:
            input("PoG single server is running but process handle is missing. Please restart the script after killing the PoG single server. Press any key to continue...")

        if check_pog_server(MULTI_ENDPOINT) and not proc_multi:
            input("PoG multi server is running but process handle is missing. Please restart the script after killing the PoG multi server. Press any key to continue...")        

        output = call_pog_api(question, topic_entities, idx, args.timeout)
        metrics = extract_metrics(output)
        elapsed = metrics.get("TIME", None)
        
        if idx % 5 == 0:
            # restart PoG to free memory
            if proc_single:
                proc_single = kill_server_safely(proc_single)
            if proc_multi:
                proc_multi = kill_server_safely(proc_multi)
            time.sleep(5)
            if check_pog_server(SINGLE_ENDPOINT) and not proc_single:
                input("PoG single server is running but process handle is missing. Please restart the script after killing the PoG single server. Press any key to continue...")
            if check_pog_server(MULTI_ENDPOINT) and not proc_multi:
                input("PoG multi server is running but process handle is missing. Please restart the script after killing the PoG multi server. Press any key to continue...")
            start_pog_server()
            # time.sleep(5)

        merge_metrics(aggregate_metrics, metrics)

        try:
            record = { "question": question, "results": output['results'], "reasoning_chains": output["reasoning_chains"], "metrics": metrics }
        except Exception as e:
            print(f"Error processing output for question index {idx}: {e}")
            print(f"Output: {output}")
            raise e
        record_full = output

        results.append(record)
        results_full.append(record_full)

        # IMPORTANT: only count successful responses for timing metrics
        if "error" not in output:
            success_count += 1
            total_elapsed += float(elapsed)
        total_elapsed_with_errors += float(elapsed) if elapsed is not None else 0.0

        metrics_summary = {k: v for k, v in aggregate_metrics.items()}
        metrics_summary["QUESTIONS"] = success_count
        metrics_summary["TIME"] = total_elapsed
        metrics_summary["TIME_PER_QUESTION"] = total_elapsed / success_count if success_count else 0
        metrics_summary["TOTAL_QUESTIONS"] = total_questions
        metrics_summary["TOTAL_TIME_WITH_ERRORS"] = total_elapsed_with_errors
        metrics_summary["TIME_PER_QUESTION_WITH_ERRORS"] = total_elapsed_with_errors / total_questions if total_questions else 0

        persist_all(base_path, results, results_full, metrics_summary)

    print(f"Saved outputs to {base_path}_*.json")


if __name__ == "__main__":
    main()
