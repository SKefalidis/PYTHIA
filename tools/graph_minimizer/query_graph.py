from functools import cache, lru_cache
import graph_tool.all as gt
import sqlite3
import time
import sys
import os
import itertools
import numpy as np
import argparse
import threading
import gc
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, jsonify, request
from scipy.stats import f
from collections import defaultdict

# Cancellation token shared across requests
_cancel_epoch = 0
_cancel_lock = threading.Lock()


def trigger_cancellation():
    """Increment the cancellation epoch to signal all active searches to stop."""
    global _cancel_epoch
    with _cancel_lock:
        _cancel_epoch += 1
        return _cancel_epoch


def current_cancellation_epoch():
    return _cancel_epoch


def was_cancelled(epoch_snapshot):
    return epoch_snapshot != _cancel_epoch

# CONFIG
GRAPH_FILE = 'graph_encoded.txt'   # Your Java output
GRAPH_CACHE = 'graph.gt'
DB_FILE = 'mapping.db'             # Created by step 1
CLASSES_FILE = 'classes.txt'       # Optional file with classes to filter

# CONSTANTS
RDF_TYPE_URI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

FREEBASE_TYPE_PREDICATE_URIS = [
    RDF_TYPE_URI,
    "http://rdf.freebase.com/ns/type.object.type",
    "http://rdf.freebase.com/ns/type.type.instance",
]

WIKIDATA_TYPE_PREDICATE_URIS = [
    RDF_TYPE_URI,
    "http://www.wikidata.org/prop/direct/P31",  # instance of
    "http://www.wikidata.org/prop/direct/P279", # subclass of
]

class GraphExplorer:
    
    def __init__(self, source_dir, filter_classes=False):
        self._connect_db()
        
        if 'freebase' in DB_FILE.lower():
            self.TYPE_PREDICATE_URIS = FREEBASE_TYPE_PREDICATE_URIS
        elif 'wikidata' in DB_FILE.lower():
            self.TYPE_PREDICATE_URIS = WIKIDATA_TYPE_PREDICATE_URIS
        else:
            self.TYPE_PREDICATE_URIS = [RDF_TYPE_URI]
        
        # Filter high-degree entities
        self.max_degree = 10000
        self.v_mask_base = None
        self.e_mask_base = None
        self.g_base = None

        # Load basic graph
        self.loaded_graph_dir = source_dir
        self.filter_classes = filter_classes
        self._load_graph(filter_classes)
        
        # Load graph with filtered type predicates
        self._type_pred_ids = None       
        self._type_pred_ids = self._type_predicates()
        self._type_predicates_filter_map = self._type_predicates_filter() 
        self.g_base_no_type_predicates = gt.GraphView(
            self.g, 
            vfilt=self.v_mask_base, 
            efilt=self._type_predicates_filter_map,
            skip_vfilt=True,
            skip_efilt=True
        )
        
    # -------------------
    # ----- Filters -----
    # -------------------

    def _type_predicates(self):
        if self._type_pred_ids is None:
            ids = []
            for uri in self.TYPE_PREDICATE_URIS:
                pid = self.get_id(uri)
                if pid is not None:
                    ids.append(pid)
            self._type_pred_ids = set(ids)
        return self._type_pred_ids
    
    def _type_predicates_filter(self):
        """Return a bool edge PropertyMap that hides all type edges."""
        type_ids = self._type_pred_ids
        if not type_ids:
            return None
        efilt = self.g.new_edge_property("bool")
        efilt.a = ~np.isin(self.g.ep.predicate.a, list(type_ids))
        return efilt
    
    def _get_class_filter_ids(self):
        """Reads classes.txt and returns a list of integer Node IDs."""
        if not os.path.exists(CLASSES_FILE):
            print(f"Warning: {CLASSES_FILE} not found. No classes will be filtered.")
            return []
        
        print(f"Reading blocked classes from {CLASSES_FILE}...")
        ids = []
        failed = 0
        with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                uri = line.strip()
                if '\t' in uri:
                    uri = uri.split('\t')[0].strip()
                if uri:
                    # Resolve URI to ID using existing DB connection
                    node_id = self.get_id(uri)
                    if node_id is not None:
                        ids.append(node_id)
                    else:
                        failed += 1
        
        print(f"  - Resolved {len(ids)} class URIs to graph IDs.")
        if failed > 0:
            print(f"  - Warning: {failed} class URIs could not be resolved to IDs.")
        return ids
    
    # --------------------------------
    # ----- Graph and DB loading -----
    # --------------------------------

    def _load_graph(self, filter_classes):
        if (os.path.exists(GRAPH_CACHE) and 
            os.path.getmtime(GRAPH_CACHE) > os.path.getmtime(GRAPH_FILE)):
            print(f"Loading cached graph from {GRAPH_CACHE}...")
            t0 = time.time()
            self.g = gt.load_graph(GRAPH_CACHE)
            t1 = time.time()
            print(f"Binary graph loaded in {t1-t0:.2f}s")
        else:
            print(f"Parsing raw data from {GRAPH_FILE}...")
            t0 = time.time()
            self.g = gt.load_graph_from_csv(
                GRAPH_FILE,
                hashed=False,
                csv_options={'quotechar': '"', 'delimiter': " "}, 
                eprop_types=['int'],      
                eprop_names=['predicate'] 
            )
            print(f"Saving binary cache to {GRAPH_CACHE}...")
            self.g.save(GRAPH_CACHE)
            t1 = time.time()
            print(f"Graph parsed and cached in {t1-t0:.2f}s")
            
        total_degrees = self.g.get_total_degrees(self.g.get_vertices())
        
        self.v_mask_base = (total_degrees < self.max_degree)
        if filter_classes:
            blocked_ids = self._get_class_filter_ids()
            if blocked_ids:
                # Update the underlying numpy array of the PropertyMap
                # Set specific indices to 0 (False) to hide them
                self.v_mask_base[blocked_ids] = 0
                print(f"  - Applied mask: {len(blocked_ids)} class vertices hidden.")
        
        self.e_mask_base = self.g.new_edge_property("bool")
        self.e_mask_base.a = np.ones(self.g.num_edges(), dtype=bool)
        self.g_base = gt.GraphView(
            self.g, 
            vfilt=self.v_mask_base, 
            efilt=self.e_mask_base, 
            skip_vfilt=False, 
            skip_efilt=False
        )

        # print(f"  Nodes: {self.g.num_vertices():,}")
        # print(f"  Edges: {self.g.num_edges():,}")
        
        # print(f"  Nodes: {self.g_base.num_vertices():,}")
        # print(f"  Edges: {self.g_base.num_edges():,}")

    def _connect_db(self):
        self.conn = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True, check_same_thread=False)
        self.cursor = self.conn.cursor()

    @lru_cache(maxsize=10000)
    def get_id(self, uri_str):
        clean_uri = uri_str.strip()
        row = self.cursor.execute("SELECT id FROM entities WHERE uri=?", (clean_uri,)).fetchone()
        return row[0] if row else None

    @lru_cache(maxsize=10000)
    def get_uri(self, node_id):
        row = self.cursor.execute("SELECT uri FROM entities WHERE id=?", (int(node_id),)).fetchone()
        return row[0] if row else f"ID_{node_id}"

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
        
        # 1. Clear LRU caches to remove references to internal objects
        self.get_id.cache_clear()
        self.get_uri.cache_clear()

        # 2. Explicitly release GraphViews (they hold refs to the main graph)
        self.g_base = None
        self.g_base_no_type_predicates = None
        
        # 3. Release filters and masks
        self.v_mask_base = None
        self.e_mask_base = None
        self._type_predicates_filter_map = None
        
        # 4. Finally release the main graph
        self.g = None
        
    # ----------------------------
    # ----- Search functions -----
    # ----------------------------
    
    def find_all_paths(self, uri_start, uri_end, max_hops=3, filter_type_predicates=False, g_clean=None, v_mask=None, epoch_snapshot=None):        
        local_cancel_epoch = current_cancellation_epoch() if epoch_snapshot is None else epoch_snapshot
        if was_cancelled(local_cancel_epoch):
            return {"error": "Search cancelled."}
        start_id = self.get_id(uri_start)
        end_id = self.get_id(uri_end)
        efilt = self._type_predicates_filter_map if filter_type_predicates else self.e_mask_base
        
        print(f"--- Fast Path Search between {uri_start} and {uri_end} ---")

        if start_id is None or end_id is None: 
            return {"error": "Error: Nodes not found."}

        if was_cancelled(local_cancel_epoch):
            return {"error": "Search cancelled."}

        # ----------------------------
        # ----- TOPOLOGY PRUNING -----
        # ----------------------------
        timer_start = time.time()

        if g_clean is None:
            # Base view using degree filter
            start_ok = self.v_mask_base[start_id]
            end_ok = self.v_mask_base[end_id]
            if start_ok and end_ok:
                print("FAST PATH triggered.")
                # FAST PATH: Both nodes are already valid. 
                v_mask = self.v_mask_base
                if filter_type_predicates:
                    g_clean = self.g_base_no_type_predicates
                else:
                    g_clean = self.g_base
            else:
                print("SLOW PATH triggered.")
                # SLOW PATH: One of the nodes is blocked.
                v_mask = self.v_mask_base.copy()
                v_mask[start_id] = True
                v_mask[end_id] = True
                g_clean = gt.GraphView(self.g, vfilt=v_mask, efilt=efilt, skip_vfilt=True, skip_efilt=True)

        # Bi-directional BFS to find corridor
        def bfs_forward():
            return gt.shortest_distance(g_clean, source=g_clean.vertex(start_id), max_dist=max_hops)
        def bfs_backward():
            return gt.shortest_distance(g_clean, source=g_clean.vertex(end_id), max_dist=max_hops)

        start = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_fwd = executor.submit(bfs_forward)
            future_bwd = executor.submit(bfs_backward)
            dist_from_start = future_fwd.result()
            dist_to_end = future_bwd.result()
        print(f"BFS searches completed in {time.time() - start:.4f}s.")

        if was_cancelled(local_cancel_epoch):
            return {"error": "Search cancelled."}
        
        # Keep nodes where: Start->Node + Node->End <= max_hops
        d_start = dist_from_start.a
        d_end = dist_to_end.a
        valid_nodes = (d_start <= max_hops) & (d_end <= max_hops) & ((d_start + d_end) <= max_hops)
        
        # Calculate intersection as numpy array first and then convert to PropertyMap
        mask_array = v_mask & valid_nodes
        final_mask = self.g.new_vertex_property("bool")
        final_mask.a = mask_array
        
        # Create corridor view
        g_corridor = gt.GraphView(self.g.base, vfilt=final_mask, efilt=efilt, skip_vfilt=True, skip_efilt=True)
        
        topology_analysis_time = time.time() - timer_start
        print(f"Topology Analysis: {topology_analysis_time:.4f}s.")

        # -------------------------
        # ----- ADJACENCY MAP -----
        # -------------------------
        t_start = time.time()

        # Get all edges in the corridor as a single NumPy array [N, 3] (source, target, predicate)
        edges_arr = g_corridor.get_edges([self.g.ep.predicate])
        
        adj = {}
        for u, v, p in edges_arr:
            if len(edges_arr) > 300_000:
                return { "error": "Corridor too large, aborting search." }
            
            if was_cancelled(local_cancel_epoch):
                return {"error": "Search cancelled."}
            
            # Forward edge (u -> v)
            if u not in adj: adj[u] = {}
            if v not in adj[u]: adj[u][v] = set()
            # Store tuple: (predicate, direction)
            adj[u][v].add((p, '->'))
            
            # Backward edge (v -> u)
            if v not in adj: adj[v] = {}
            if u not in adj[v]: adj[v][u] = set()
            # Store tuple: (predicate, direction)
            adj[v][u].add((p, '<-'))
            
        adjacency_time = time.time() - t_start
        print(f"Adjacency Prep: {adjacency_time:.4f}s. Edges in corridor: {len(edges_arr)}")
        
        if was_cancelled(local_cancel_epoch):
            return {"error": "Search cancelled."}

        if len(edges_arr) > 300_000:
            return { "error": "Corridor too large, aborting search." }

        # -----------------------------
        # ----- # PATH GENERATION -----
        # -----------------------------
        t_start = time.time()
        
        t_timer = 0
        t_remaining = max(30, 60-adjacency_time - topology_analysis_time)
        
        # Use deque for better pop(0) performance if needed, but stack (LIFO) is fine here
        # Optimization: use frozenset for visited to enable fast hashing
        stack = [(start_id, (), (), frozenset([start_id]))]
        chain_counts = defaultdict(int)
        check_counter = 0
        while stack:
            t_timer = time.time() - t_start
            if was_cancelled(local_cancel_epoch):
                return {"error": "Search cancelled."}
            check_counter += 1
            if check_counter % 10000 == 0:
                if t_timer > t_remaining:
                    print(f"Path expansion time limit reached ({t_remaining}s). Stopping search.")
                    return {
                        "error": "Path expansion time limit reached.",
                    }
            u, p_chain, d_chain, visited = stack.pop()
            # If we popped the target (rarely reached if optimization below is used, but good safety)
            if u == end_id and len(p_chain) > 0:
                chain_counts[(p_chain, d_chain)] += 1
                continue
            # Pruning: If adding 1 more hop exceeds max, stop
            if len(p_chain) >= max_hops:
                continue
            # Expand
            if u in adj:
                neighbors = adj[u]
                for v, edge_infos in neighbors.items():
                    # edge_infos is a set of tuples: {(pred1, '->'), (pred2, '<-'), ...}
                    
                    if v not in visited:
                        if v == end_id:
                            # Target found: Add all connecting edges to results
                            for p, d in edge_infos:
                                chain_counts[(p_chain + (p,), d_chain + (d,))] += 1
                        elif len(p_chain) + 1 < max_hops:
                            # Continue search
                            new_visited = visited | {v}
                            for p, d in edge_infos:
                                stack.append((v, p_chain + (p,), d_chain + (d,), new_visited))

        path_expansion_time = time.time() - t_start
        print(f"Path Expansion: {path_expansion_time:.4f}s.")

        # ----------------------
        # ----- FORMATTING -----
        # ----------------------
        start_time = time.time()
        
        # Sort by length
        sorted_chains = sorted(chain_counts.items(), key=lambda x: len(x[0][0]))
        
        output = []
        predicates_lists = []
        path_lengths = []
        directions_lists = []
        counts_list = [] # New list for counts
    
        for (chain_p, chain_d), count in sorted_chains:
            predicates_list = []
            directions_list = []
            predicate_path = uri_start
            placeholder_index = 0
            
            for pid, direction in zip(chain_p, chain_d):
                if placeholder_index > 0:
                    predicate_path += f"x{placeholder_index}"
                if direction == '->':
                    predicate_path += f" -- {self.get_uri(pid)} --> "
                else:
                    predicate_path += f" <-- {self.get_uri(pid)} -- "
                placeholder_index += 1
                predicates_list.append(self.get_uri(pid))
                directions_list.append(direction)
                
            predicate_path += uri_end
            # CHANGE 3: Include count in output string
            output.append(f"[{len(output)+1} (len: {len(chain_p)})] ({count} matches) {predicate_path}")
            
            predicates_lists.append(predicates_list)
            directions_lists.append(directions_list)
            path_lengths.append(len(chain_p))
            counts_list.append(count)
        
        formatting_time = time.time() - start_time
        print(f"Formatting: {formatting_time:.4f}s.")
            
        print(f"\n--- Found {len(output)} unique predicate chains in {(topology_analysis_time + adjacency_time + path_expansion_time + formatting_time):.4f}s ---")        
        
        for line in output:
            print(line)

        return {
            "paths": output,
            "predicates": predicates_lists,
            "directions": directions_lists,
            "path_lengths": path_lengths,
            "counts": counts_list,
            "topology_analysis_time": topology_analysis_time,
            "adjacency_time": adjacency_time,
            "path_expansion_time": path_expansion_time,
            "formatting_time": formatting_time,
        }
        
    def pythia_find_all_paths(self, uri_start, uri_end, filter_type_predicates=False, additional_hops=1):
        local_cancel_epoch = current_cancellation_epoch()
        start_id = self.get_id(uri_start)
        end_id = self.get_id(uri_end)
        efilt = self._type_predicates_filter_map if filter_type_predicates else self.e_mask_base
        
        print(f"--- Fast Path Search between {uri_start} and {uri_end} ---")

        if start_id is None or end_id is None: 
            return "Error: Nodes not found."

        if was_cancelled(local_cancel_epoch):
            return "Search cancelled."

        # ----------------------------
        # ----- TOPOLOGY PRUNING -----
        # ----------------------------        
        # Base view using degree filter
        start_ok = self.v_mask_base[start_id]
        end_ok = self.v_mask_base[end_id]

        if start_ok and end_ok:
            print("FAST PATH triggered.")
            # FAST PATH: Both nodes are already valid. 
            v_mask = self.v_mask_base
            if filter_type_predicates:
                g_clean = self.g_base_no_type_predicates
            else:
                g_clean = self.g_base
        else:
            print("SLOW PATH triggered.")
            # SLOW PATH: One of the nodes is blocked.
            v_mask = self.v_mask_base.copy()
            v_mask[start_id] = True
            v_mask[end_id] = True
            g_clean = gt.GraphView(self.g, vfilt=v_mask, efilt=efilt, skip_vfilt=True, skip_efilt=True)

        shortest_distance = gt.shortest_distance(g_clean, source=g_clean.vertex(start_id), target=g_clean.vertex(end_id))
        
        start_distance = max(2, int(shortest_distance)) # paths of length 1 will be found anyway :^)
        
        results = self.find_all_paths(
            uri_start,
            uri_end,
            max_hops=start_distance + additional_hops,
            filter_type_predicates=filter_type_predicates,
            g_clean=g_clean,
            v_mask=v_mask,
            epoch_snapshot=local_cancel_epoch,
        )
        
        if "error" in results:
            return results["error"]

        paths = results.get("paths")
        predicates = results.get("predicates")
        directions = results.get("directions")
        path_lengths = results.get("path_lengths")
        counts = results.get("counts")
        return {
            "paths": paths,
            "predicates": predicates,
            "directions": directions,
            "path_lengths": path_lengths,
            "counts": counts,
        }

# -------------------------
# ----- Graph Loading -----
# -------------------------

def _configure_paths(source_dir):
    global GRAPH_FILE, GRAPH_CACHE, DB_FILE, CLASSES_FILE
    GRAPH_FILE = os.path.join(source_dir, 'graph_encoded.txt')
    GRAPH_CACHE = os.path.join(source_dir, 'graph.gt')
    DB_FILE = os.path.join(source_dir, 'mapping.db')
    CLASSES_FILE = os.path.join(source_dir, 'classes.txt')


def _ensure_inputs_exist():
    if not (os.path.exists(GRAPH_FILE) or os.path.exists(GRAPH_CACHE)):
        raise FileNotFoundError(f"Graph file not found: {GRAPH_FILE}")
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"DB file not found: {DB_FILE}")

explorer: GraphExplorer = None
explorer_lock = threading.Lock()

def load_graph(source_dir, filter_classes=False):
    global explorer
    if './' not in source_dir[:2]:
        source_dir = './' + source_dir

    with explorer_lock:
        if explorer is not None:
            if explorer.loaded_graph_dir == source_dir and explorer.filter_classes == filter_classes:
                return {
                    "graph_file": GRAPH_FILE,
                    "graph_cache": GRAPH_CACHE,
                    "db_file": DB_FILE,
                    "nodes": explorer.g.num_vertices() if explorer.g else 0,
                    "edges": explorer.g.num_edges() if explorer.g else 0,
                }
            else:
                explorer.close()
                del explorer
                explorer = None
                gc.collect()
                
        _configure_paths(source_dir)
        _ensure_inputs_exist()
        explorer = GraphExplorer(source_dir, filter_classes=filter_classes)
        return {
            "graph_file": GRAPH_FILE,
            "graph_cache": GRAPH_CACHE,
            "db_file": DB_FILE,
            "nodes": explorer.g.num_vertices() if explorer.g else 0,
            "edges": explorer.g.num_edges() if explorer.g else 0,
        }

# ----------------------
# ----- Web Server -----
# ----------------------

app = Flask(__name__)

@app.route("/load-graph", methods=["POST"])
def load_graph_route():
    data = request.get_json(silent=True) or {}
    source_dir = data.get("source_dir")
    filter_classes = data.get("filter_classes")
    if not source_dir:
        return jsonify({"status": "error", "error": "source_dir is required"}), 400

    try:
        meta = load_graph(source_dir, filter_classes=bool(int(filter_classes)))
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400

    return jsonify({"status": "ok", "meta": meta})


@app.route("/all-paths", methods=["POST"])
def graph_search_route():
    data = request.get_json(silent=True) or {}
    uri_start = data.get("uri_start")
    uri_end = data.get("uri_end")
    max_hops = data.get("max_hops")
    filter_type_predicates = data.get("fp")

    if not uri_start or not uri_end:
        return jsonify({"status": "error", "error": "uri_start and uri_end are required"}), 400

    try:
        max_hops_int = int(max_hops)
    except Exception:
        return jsonify({"status": "error", "error": "max_hops must be an integer"}), 400
    
    try:
        filter_type_predicates = bool(int(filter_type_predicates))
    except Exception:
        return jsonify({"status": "error", "error": "fp must be set to either 0 or 1"}), 400

    with explorer_lock:
        if explorer is None:
            return jsonify({"status": "error", "error": "No graph loaded. Call /load-graph first."}), 400
        result = explorer.find_all_paths(uri_start, uri_end, max_hops=max_hops_int, filter_type_predicates=filter_type_predicates)

    return jsonify({"status": "ok", "result": result})

@app.route("/pythia-all-paths", methods=["POST"])
def pythia_graph_search_route():
    data = request.get_json(silent=True) or {}
    uri_start = data.get("uri_start")
    uri_end = data.get("uri_end")
    filter_type_predicates = data.get("fp")
    additional_hops = data.get("additional_hops", 1)

    if not uri_start or not uri_end:
        return jsonify({"status": "error", "error": "uri_start and uri_end are required"}), 400

    try:
        filter_type_predicates = bool(int(filter_type_predicates))
    except Exception:
        return jsonify({"status": "error", "error": "fp must be set to either 0 or 1"}), 400

    with explorer_lock:
        if explorer is None:
            return jsonify({"status": "error", "error": "No graph loaded. Call /load-graph first."}), 400
        result = explorer.pythia_find_all_paths(uri_start, uri_end, filter_type_predicates=filter_type_predicates, additional_hops=additional_hops)

    return jsonify({"status": "ok", "result": result})


@app.route("/cancel", methods=["POST"])
def cancel_requests_route():
    epoch = trigger_cancellation()
    return jsonify({"status": "ok", "message": "Cancellation signaled", "epoch": epoch})

# -----------------------
# ----- Application -----
# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Path Explorer (web server or CLI).")
    parser.add_argument("--source-dir", type=str, help="Directory containing graph and database files.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the web server (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=65023, help="Port for the web server (default: 65023).")
    parser.add_argument("--cli", action="store_true", help="Run interactive CLI instead of the web server.")
    args = parser.parse_args()

    if args.cli:
        if not args.source_dir:
            print("Error: --source-dir is required in CLI mode.")
            sys.exit(1)

        meta = load_graph(args.source_dir)
        print(f"Loaded graph: {meta['nodes']:,} nodes, {meta['edges']:,} edges")
        print("\nSystem Ready.")
        print("Usage: StartURI, EndURI, [hops]")

        while True:
            try:
                user_input = input("\nEnter: ").strip()
                if not user_input: continue
                if user_input.lower() in ['exit', 'quit']: break

                parts = [x.strip() for x in user_input.split(',')]
                start = parts[0]
                end = parts[1]
                hops = int(parts[2]) if len(parts) > 2 and parts[2] else 3

                print(explorer.find_all_paths(start, end, max_hops=hops))

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        if args.source_dir:
            try:
                meta = load_graph(args.source_dir)
                print(f"Preloaded graph: {meta['nodes']:,} nodes, {meta['edges']:,} edges")
            except Exception as exc:
                print(f"Error loading initial graph: {exc}")
                sys.exit(1)

        print(f"Starting web server on {args.host}:{args.port} ...")
        app.run(host=args.host, port=args.port)