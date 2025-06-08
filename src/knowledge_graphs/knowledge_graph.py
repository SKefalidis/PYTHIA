from functools import lru_cache
import sqlite3
import re
import sys
import os
import csv
import time
from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Optional, Dict

# Assuming these exist in your project structure
from src.engine.config import CONFIG
from src.utils import is_uri, read_configuration, execute_sparql_query

from src.logging import log, LogComponent, LogLevel, LogType
from src.metrics import get_kgaqa_tracker


# ------------------------
# ----- ENUMERATIONS -----
# ------------------------

class Direction(Enum):
    OUTGOING = "outgoing" # normal
    INCOMING = "incoming" # reverse
    UNKNOWN = "unknown"
    ROOT = "root"

class KgComponentType(Enum):
    ENTITY = "entity"
    CLASS = "class"
    PREDICATE = "predicate"

class StartPoint(Enum):
    CLASS = "class" 
    ENTITY = "entity" 

# ------------------------
# ----- DATA CLASSES -----
# ------------------------

class KgComponent:
    """
    A lightweight wrapper around a database row. 
    Attributes are fetched lazily or stored minimally.
    """
    __slots__ = ("_db", "_id", "type", "label", "description", 
                 "outgoing_edges_count", "incoming_edges_count")

    def __init__(self, db_conn: sqlite3.Connection, row_id, type_val, label, desc, out_c, in_c):
        self._db = db_conn
        self._id = row_id
        self.type = KgComponentType(type_val)
        self.label = label
        self.description = desc
        self.outgoing_edges_count = out_c
        self.incoming_edges_count = in_c

    @property
    def uri(self) -> str:
        # Fetch string only when asked
        cursor = self._db.execute("SELECT value FROM strings WHERE id = ?", (self._id,))
        return cursor.fetchone()[0]

    @property
    def parent_classes(self) -> List[str]:
        # Fetch parents on demand
        query = """
            SELECT s.value FROM node_parents np
            JOIN strings s ON np.parent_id = s.id
            WHERE np.node_id = ?
        """
        cursor = self._db.execute(query, (self._id,))
        return [row[0] for row in cursor.fetchall()]
    
    def is_class(self) -> bool:
        return self.type == KgComponentType.CLASS
    
    def is_entity(self) -> bool:
        return self.type == KgComponentType.ENTITY
    
    def is_predicate(self) -> bool:
        return self.type == KgComponentType.PREDICATE
    
    def __repr__(self):
        return f"KgComponent(uri={self.uri}, type={self.type}, label={self.label}, description={self.description}, outgoing_edges_count={self.outgoing_edges_count}, incoming_edges_count={self.incoming_edges_count}, parents={self.parent_classes})"

class PredicateInfo:
    
    def __init__(self, uri: str, direction: Direction, cardinality: int, objects: List[str] = []):
        self.uri = uri
        self.direction = direction
        self.cardinality = cardinality
        self.objects = objects # URIs or literals
        if isinstance(direction, str):
            if direction == "outgoing":
                self.direction = Direction.OUTGOING
            elif direction == "incoming":
                self.direction = Direction.INCOMING
            elif direction == "unknown":
                self.direction = Direction.UNKNOWN
            else:
                raise ValueError(f"Invalid direction: {direction}")
            
    def __repr__(self):
        return f"PredicateInfo(uri={self.uri}, direction={self.direction}, cardinality={self.cardinality}, objects={self.objects[:3]}...)"
    
    def get_description(self, readable: bool = True, kg: 'KnowledgeGraph' = None) -> str:
        self_str = self.get_label(kg) if readable else self.uri
        objects = self.objects[:3] if len(self.objects) > 3 else self.objects
        objects_str = ", ".join([kg.get_kg_component(o).label if (readable and kg and kg.get_kg_component(o)) else o for o in objects])
        if len(self.objects) > 3:
            cardinality_str = f"(and {len(self.objects) - 3} more)"
        elif len(self.objects) > 0:
            cardinality_str = f""
        else:
            cardinality_str = f"({self.cardinality} matches)"

        if self.direction == Direction.OUTGOING:
            return f"[{self_str}] to [{objects_str} {cardinality_str}]"
        elif self.direction == Direction.INCOMING:
            return f"[{self_str}] from [{objects_str} {cardinality_str}]"
        elif self.direction == Direction.UNKNOWN:
            return f"[{self_str}] unknown direction [{objects_str} {cardinality_str}]"
        
    def get_label(self, kg: 'KnowledgeGraph') -> str:
        kg_component = kg.get_kg_component(self.uri)
        if kg_component is not None:
            return kg_component.label
        else:
            return self.uri
        
    def get_direction_arrow(self) -> str:
        if self.direction == Direction.OUTGOING:
            return "->"
        elif self.direction == Direction.INCOMING:
            return "<-"
        else:
            return "unknown"
        
    def get_direction_word(self) -> str:
        if self.direction == Direction.OUTGOING:
            return "to"
        elif self.direction == Direction.INCOMING:
            return "from"
        else:
            return "unknown"
        
    def get_objects_string(self, readable: bool = True, kg: 'KnowledgeGraph' = None, with_uri: bool = False) -> str:
        objects = self.objects[:3] if len(self.objects) > 3 else self.objects
        # objects_str = ", ".join([kg.get_kg_component(o).label if (readable and kg and kg.get_kg_component(o)) else o for o in objects])
        objects_str = ""
        for o in objects:
            if readable and kg:
                kgc = kg.get_kg_component(o)
                if kgc:
                    objects_str += kgc.label + ", "
                else:
                    if "http://" in o:
                        objects_str += "Unnamed/Intermediate Node "
                        if with_uri:
                            short_uri = kg.shorten_uri(o)
                            objects_str += f"({short_uri})"
                    else:
                        objects_str += o
                    objects_str += ", "
            else:
                objects_str += o + ", "
        if len(self.objects) > 3:
            cardinality_str = f"(and {len(self.objects) - 3} more)"
        elif len(self.objects) > 0:
            cardinality_str = None
        else:
            cardinality_str = f"({self.cardinality} matches)"
        if cardinality_str is None:
            return f"{objects_str.rstrip(', ')}"
        else:
            return f"{objects_str.rstrip(', ')} {cardinality_str}"
        
    def get_objects_sample_type(self) -> str:
        objects = self.objects[:3] if len(self.objects) > 3 else self.objects
        if all(is_uri(o) for o in objects):
            return "URI"
        elif all(not is_uri(o) for o in objects):
            return "Literal"
        else:
            return "Mixed"
        
class PredicateInfoPath:

    def __init__(self, start_uri: str, predicate_infos: List[PredicateInfo]):
        self.start_uri = start_uri
        self.predicate_infos = predicate_infos
        
    # def verbalize_predicate_path(self, readable: bool = True, kg: 'KnowledgeGraph' = None) -> str:
    #     verbalization = ""
    #     for pi in self.predicate_infos:
    #         verbalization += pi.get_label(kg) if readable else pi.uri
    #         verbalization += " -> "
    #     verbalization = verbalization.rstrip(" -> ")
    #     return verbalization
        
    def verbalize_path(self, readable: bool = True, kg: 'KnowledgeGraph' = None) -> str:
        verbalization = ""
        for pi in self.predicate_infos:
            verbalization += pi.get_description(readable, kg) + "."
        return verbalization

class KgEntity(KgComponent):
    """
    Extension of KgComponent for entities that fetches predicate details on demand.
    """
    
    def get_predicates(self, sparql_endpoint: str, filter_literals: bool, limit: int = 0) -> List[PredicateInfo]:
        return self.get_predicates_for_entity(self.uri, sparql_endpoint, filter_literals, limit)
    
    @classmethod
    @lru_cache(maxsize=256)
    def get_predicates_for_entity(cls, uri: str, sparql_endpoint: str, filter_literals: bool, limit: int = 0) -> List[PredicateInfo]:
        start_time = time.time()
        if not filter_literals:
            query = f"""
            SELECT DISTINCT ?direction ?p ?o
            WHERE {{
                {{
                    <{uri}> ?p ?o .
                    BIND("outgoing" AS ?direction)
                }}
                UNION
                {{
                    ?o ?p <{uri}> .
                    BIND("incoming" AS ?direction)
                }}
            }}
            """
            if limit > 0:
                query += f"LIMIT {limit}"
        else:
            query = f"""
            SELECT DISTINCT ?direction ?p ?o
            WHERE {{
                {{
                    <{uri}> ?p ?o .
                    FILTER (!isLiteral(?o))
                    BIND("outgoing" AS ?direction)
                }}
                UNION
                {{
                    ?o ?p <{uri}> .
                    BIND("incoming" AS ?direction)
                }}
            }}
            """
            if limit > 0:
                query += f"LIMIT {limit}"
        # print(query)
        log(f"distinct predicates for entity {uri}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG, LogType.HEADER)
        log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
        
        try:
            print("Executing SPARQL query to get distinct predicates with limit =", limit)
            query_result = execute_sparql_query(query, sparql_endpoint)
            results = query_result.convert()
        except Exception as e:
            print("Error executing SPARQL query:", e)
            log(f"Error get_distinct_predicates_for_entity: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.ERROR)
            log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.ERROR)
            get_kgaqa_tracker()._get_distinct_predicates_for_entity_calls += 1
            get_kgaqa_tracker()._get_distinct_predicates_for_entity_time += time.time() - start_time
            return []
        
        directions: List[Direction] = []
        predicates: List[str] = []
        objects: List[str] = []
        for result in results["results"]["bindings"]:
            # print(result)
            direction = result["direction"]["value"]
            directions.append(Direction.OUTGOING if direction == "outgoing" else Direction.INCOMING)
            predicates.append(result["p"]["value"])
            objects.append(result["o"]["value"])

        # gather information
        direction_predicate_to_objects: Dict[str, List[str]] = {}
        for d, p, o in zip(directions, predicates, objects):
            key = d.value + "|" + p
            if key not in direction_predicate_to_objects:
                direction_predicate_to_objects[key] = []
            direction_predicate_to_objects[key].append(o)
            
        # build PredicateInfo list
        predicate_infos: List[PredicateInfo] = []
        for key, objs in direction_predicate_to_objects.items():
            d_str, p = key.split("|")
            d_enum = Direction.OUTGOING if d_str == "outgoing" else Direction.INCOMING
            predicate_infos.append(PredicateInfo(p, d_enum, len(objs), objs))
        
        get_kgaqa_tracker()._get_distinct_predicates_for_entity_calls += 1
        get_kgaqa_tracker()._get_distinct_predicates_for_entity_time += time.time() - start_time
        return predicate_infos

class KgClass(KgComponent):
    """
    Extension of KgComponent that fetches predicate details on demand.
    """

    def _fetch_predicates(self, direction_enum: Direction, start_enum: StartPoint, leads_to_uri: bool = False) -> List[PredicateInfo]:
        # 1=Outgoing, 2=Incoming / 1=Class, 2=Entity
        d_val = 1 if direction_enum == Direction.OUTGOING else 2
        s_val = 1 if start_enum == StartPoint.CLASS else 2
        
        query = """
            SELECT s.value, np.cardinality 
            FROM node_predicates np
            JOIN strings s ON np.predicate_id = s.id
            WHERE np.node_id = ? AND np.direction = ? AND np.start_point = ? AND np.leads_to_uri = ?
        """
        cursor = self._db.execute(query, (self._id, d_val, s_val, leads_to_uri))
        return [PredicateInfo(r[0], direction_enum, r[1], []) for r in cursor.fetchall()]

    # --- Properties mimicking the old list logic ---
    
    @property
    def incoming_predicates(self) -> List[PredicateInfo]:
        return self._fetch_predicates(Direction.INCOMING, StartPoint.CLASS)
    
    @property
    def incoming_predicates_from_instances(self) -> List[PredicateInfo]:
        return self._fetch_predicates(Direction.INCOMING, StartPoint.ENTITY)
    
    @property
    def outgoing_predicates(self) -> List[PredicateInfo]:
        return self._fetch_predicates(Direction.OUTGOING, StartPoint.CLASS)
    
    @property
    def outgoing_predicates_from_instances(self) -> List[PredicateInfo]:
        return self._fetch_predicates(Direction.OUTGOING, StartPoint.ENTITY)
    
    @property
    def outgoing_predicates_that_lead_to_uris(self) -> List[PredicateInfo]:
        return self._fetch_predicates(Direction.OUTGOING, StartPoint.CLASS, leads_to_uri=True)
    
    @property
    def outgoing_predicates_that_lead_to_uris_from_instances(self) -> List[PredicateInfo]:
        return self._fetch_predicates(Direction.OUTGOING, StartPoint.ENTITY, leads_to_uri=True)
    
    def get_own_predicates(self) -> List[PredicateInfo]:
        # Helper to fetch own class predicates only
        all_p = []
        all_p.extend(self.outgoing_predicates)
        all_p.extend(self.incoming_predicates)
        return all_p
    
    def get_entity_predicates(self) -> List[PredicateInfo]:
        # Helper to fetch instance-based predicates only
        all_p = []
        all_p.extend(self.outgoing_predicates_from_instances)
        all_p.extend(self.incoming_predicates_from_instances)
        return all_p

    def get_all_predicates(self) -> List[PredicateInfo]:
        # Helper to fetch all variations
        all_p = []
        all_p.extend(self.outgoing_predicates)
        all_p.extend(self.outgoing_predicates_from_instances)
        all_p.extend(self.incoming_predicates)
        all_p.extend(self.incoming_predicates_from_instances)
        return all_p

    def get_predicates(self, direction: Direction, start: StartPoint) -> List[PredicateInfo]:
        return self._fetch_predicates(direction, start)

    def get_predicate_uris(self, direction: Direction, start: StartPoint) -> List[str]:
        return [p.uri for p in self.get_predicates(direction, start)]

# ---------------------------
# ----- Knowledge Graph -----
# ---------------------------

class KnowledgeGraph(ABC):
    
    def __init__(self):
        self.config = read_configuration("_config.yaml")
        self.db_path = None
        self.conn = None
        self.index_path = None
        
        self._class_cache = {}
        self._entity_cache = {}
        
        self.uri_to_uril_map = {}
        self.uril_to_uri_map = {}
        
    def __del__(self):
        self._class_cache.clear()
        self._entity_cache.clear()
        self.unload()

    def _connect_db(self):
        if not self.db_path:
            raise Exception("Database path not set.")
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Performance Tuning
        self.conn.execute("PRAGMA journal_mode = WAL") # Write-Ahead Log for concurrency
        self.conn.execute("PRAGMA synchronous = NORMAL")
        # MEMORY MAP: 4GB limit (Adjust as needed)
        self.conn.execute("PRAGMA mmap_size = 4294967296") 
        self.conn.execute("PRAGMA cache_size = -200000") # 200MB Page Cache

    def _init_schema(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS strings (id INTEGER PRIMARY KEY, value TEXT UNIQUE)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_strings_val ON strings(value)")
        
        # type stored as string or int. Using string for now to match your Enums.
        c.execute("""CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY, 
            type TEXT, 
            label TEXT, 
            description TEXT, 
            out_c INTEGER, 
            in_c INTEGER
        )""")
        
        c.execute("CREATE TABLE IF NOT EXISTS node_parents (node_id INTEGER, parent_id INTEGER, PRIMARY KEY(node_id, parent_id)) WITHOUT ROWID")
        
        # Direction: 1=Out, 2=In. Start: 1=Class, 2=Entity. LeadsUri: 0/1
        c.execute("""CREATE TABLE IF NOT EXISTS node_predicates (
            node_id INTEGER, 
            predicate_id INTEGER, 
            direction INTEGER, 
            start_point INTEGER, 
            cardinality INTEGER, 
            leads_to_uri INTEGER,
            PRIMARY KEY (node_id, predicate_id, direction, start_point, leads_to_uri)
        ) WITHOUT ROWID""")

        c.execute("CREATE TABLE IF NOT EXISTS uril_mappings (uri_id INTEGER PRIMARY KEY, uril_id INTEGER)")
        self.conn.commit()

    @property
    def endpoint(self):
        endpoint_server: str|None = CONFIG().get("endpoint_server")
        if endpoint_server == None:
            raise ValueError("Endpoint server not configured. Please set 'endpoint_server' in the configuration.")
        if endpoint_server.startswith("http://") or endpoint_server.startswith("https://"):
            return endpoint_server
        if self.__class__.__name__.lower() in self.config: # use default endpoint from KG config if available
            return f"http://{endpoint_server}/repositories/" + self.config[self.__class__.__name__.lower()]['default_graphdb_endpoint']
        raise ValueError(f"Invalid endpoint server configuration: {endpoint_server}")
        
    @property
    def prefixes(self):
        raise NotImplementedError("Subclasses must implement prefixes property.")
    
    @property
    def prefixes_map(self) -> Dict[str, str]:
        prefix_map = {}
        for line in self.prefixes.strip().split("\n"):
            match = re.match(r'PREFIX\s+(\w+):\s+<([^>]+)>', line)
            if match:
                prefix, uri = match.groups()
                prefix_map[prefix] = uri
        return prefix_map
    
    @property
    # @abstractmethod
    def entity_popularity_k(self) -> int:
        "Normally set to the median, we set it higher to bias towards popular entities."
        raise NotImplementedError("Subclasses must implement entity_popularity_k property.")
    
    @property
    # @abstractmethod
    def class_popularity_k(self) -> int:
        raise NotImplementedError("Subclasses must implement class_popularity_k property.")
    
    @property
    # @abstractmethod
    def predicate_popularity_k(self) -> int:
        raise NotImplementedError("Subclasses must implement predicate_popularity_k property.")
    
    # --- ACCESSORS ---

    def get_kg_component(self, uri: str) -> Optional[KgComponent]:
        if not self.conn:
            raise Exception("Database not connected. Have you loaded the knowledge graph?")
        cursor = self.conn.execute("SELECT id FROM strings WHERE value = ?", (uri,))
        res = cursor.fetchone()
        if not res: return None
        uri_id = res[0]
        
        cursor.execute("SELECT type, label, description, out_c, in_c FROM nodes WHERE id = ?", (uri_id,))
        row = cursor.fetchone()
        if not row: return None
        
        type_str, label, desc, out_c, in_c = row
        
        if type_str == KgComponentType.CLASS.value:
            return KgClass(self.conn, uri_id, type_str, label, desc, out_c, in_c)
        elif type_str == KgComponentType.ENTITY.value:
            return KgEntity(self.conn, uri_id, type_str, label, desc, out_c, in_c)
        elif type_str == KgComponentType.PREDICATE.value:
            return KgComponent(self.conn, uri_id, type_str, label, desc, out_c, in_c)
        else:
            raise ValueError(f"Unknown KgComponentType: {type_str} for URI: {uri}")
        
    def is_class(self, node: str):
        if node in self._class_cache:
            return self._class_cache[node]
            
        kgc = self.get_kg_component(node)
        if kgc is None:
            log(f"Could not retrieve knowledge graph component for node: {node}", LogComponent.OTHER, LogLevel.WARNING)
            return None
            # raise ValueError(f"Could not retrieve knowledge graph component for node: {node}")
        
        result = kgc.is_class()
        self._class_cache[node] = result
            
        return result
    
    def is_entity(self, node: str):
        if node in self._entity_cache:
            return self._entity_cache[node]

        kgc = self.get_kg_component(node)
        if kgc is None:
            log(f"Could not retrieve knowledge graph component for node: {node}", LogComponent.OTHER, LogLevel.WARNING)
            return None
        
        result = kgc.is_entity()
        self._entity_cache[node] = result
            
        return result
    
    def get_label(self, uri: str) -> Optional[str]:
        kgc = self.get_kg_component(uri)
        if kgc is None:
            return None
        return kgc.label
    
    # --- LOADING LOGIC ---

    def load(self, knowledge_graph_index_path: str):
        # Connect immediately to enable lookups
        print(f"Loading knowledge graph from index at: {knowledge_graph_index_path}")
        self.index_path = knowledge_graph_index_path
        self.db_path = os.path.join(knowledge_graph_index_path, "kg_database.sqlite3")
        
        # 1. Check if DB exists and has data
        self._connect_db()
        try:
            count = self.conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
            if count > 0:
                print("Knowledge graph already loaded.")
                return
        except sqlite3.OperationalError:
            pass # Tables don't exist yet

        print("--- STARTING HIGH-PERFORMANCE IMPORT ---")
        start_time = time.time()
        
        # 2. DANGEROUSLY FAST SETTINGS (Only for import)
        # Turn off disk synchronization and rollback journals. 
        # If the script crashes here, the DB will be corrupt, but we just delete and restart.
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA journal_mode = OFF") 
        self.conn.execute("PRAGMA cache_size = -4000000") # Use up to 4GB RAM for cache
        self.conn.execute("PRAGMA locking_mode = EXCLUSIVE")
        self.conn.execute("PRAGMA temp_store = MEMORY")
        
        self._import_data_optimized(knowledge_graph_index_path)
        
        # 3. RESTORE SAFETY
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA locking_mode = NORMAL")
        
        print(f"--- IMPORT FINISHED in {time.time() - start_time:.2f} seconds ---")
        
    def unload(self):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass

    def _import_data_optimized(self, path: str):
        c = self.conn.cursor()
        
        # --- PHASE 1: INIT TABLES (NO INDICES YET) ---
        print("[1/5] Creating Tables...")
        c.execute("DROP TABLE IF EXISTS strings")
        c.execute("DROP TABLE IF EXISTS nodes")
        c.execute("DROP TABLE IF EXISTS node_parents")
        c.execute("DROP TABLE IF EXISTS node_predicates")
        c.execute("DROP TABLE IF EXISTS uril_mappings")

        c.execute("CREATE TABLE strings (id INTEGER PRIMARY KEY, value TEXT)")
        # Note: No UNIQUE index yet, we handle uniqueness in Python to save DB work
        
        c.execute("""CREATE TABLE nodes (
            id INTEGER PRIMARY KEY, 
            type TEXT, 
            label TEXT, 
            description TEXT, 
            out_c INTEGER, 
            in_c INTEGER
        )""") # PK is technically an index, but unavoidable
        
        c.execute("CREATE TABLE node_parents (node_id INTEGER, parent_id INTEGER)")
        c.execute("CREATE TABLE node_predicates (node_id INTEGER, predicate_id INTEGER, direction INTEGER, start_point INTEGER, cardinality INTEGER, leads_to_uri INTEGER)")
        c.execute("CREATE TABLE uril_mappings (uri_id INTEGER PRIMARY KEY, uril_id INTEGER)")
        
        # --- PHASE 2: HARVEST STRINGS (Pass 1) ---
        # We read all files just to find strings. This allows us to assign IDs *before* inserting nodes.
        print("[2/5] Harvesting Strings (Memory Intensive)...")
        unique_strings = set()
        
        tsv_path = os.path.join(path, "all.tsv")
        
        # Helper to read TSV safely
        def valid_rows(p, min_cols):
            if not os.path.exists(p): return
            with open(p, "r", encoding="utf-8") as f:
                csv.field_size_limit(sys.maxsize)
                reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar='\\')
                for r in reader:
                    if len(r) >= min_cols: yield r

        # Scan nodes file
        progress = 0
        print("      Scanning nodes...")
        for row in valid_rows(tsv_path, 6):
            unique_strings.add(row[0]) # URI
            if len(row) > 6 and row[6]:
                unique_strings.update(row[6].split("|")) # Parents
            progress += 1
            if progress % 500000 == 0:
                print(f"      Processed {progress:,} nodes...", end="\r")

        # Scan predicates files
        progress = 0
        print(f"    Scanning predicates...")
        for fname in ["classes_predicates.tsv", "classes_predicates_no_literals.tsv"]:
            for row in valid_rows(os.path.join(path, fname), 4):
                if row[4] != "0":
                    unique_strings.add(row[0]) # Class
                    unique_strings.add(row[3]) # Predicate
                progress += 1
                if progress % 500000 == 0:
                    print(f"      Processed {progress:,} predicates...", end="\r")

        print(f"      Found {len(unique_strings):,} unique strings.")
        
        # Bulk Insert Strings using Generator
        # This converts set -> [(s,), (s,)...] on the fly
        print("      Inserting Strings...")
        c.executemany("INSERT INTO strings (value) VALUES (?)", ((s,) for s in unique_strings))
        
        # Create Index NOW so we can map them back fast
        print("      Indexing Strings...")
        c.execute("CREATE UNIQUE INDEX idx_strings_val ON strings(value)")
        
        # --- PHASE 3: BUILD MEMORY MAP ---
        print("[3/5] Building ID Map...")
        # Load {string: id} into RAM. This allows super-fast lookups during data insert.
        c.execute("SELECT value, id FROM strings")
        str_map = {row[0]: row[1] for row in c.fetchall()}
        
        # --- PHASE 4: INSERT DATA (Pass 2) ---
        print("[4/5] Bulk Inserting Data...")
        
        # A. Nodes & Parents
        batch_nodes = []
        batch_parents = []
        BATCH_SIZE = 50000 
        
        for row in valid_rows(tsv_path, 6):
            uri, type_str, label, desc, out_c, in_c = row[0], row[1], row[2], row[3], row[4], row[5]
            
            # Fast lookup
            if uri not in str_map: continue # Should not happen
            uid = str_map[uri]
            
            try:
                batch_nodes.append((uid, type_str, label, desc, int(out_c), int(in_c)))
                
                if len(row) > 6 and row[6]:
                    for p in row[6].split("|"):
                        if p in str_map:
                            batch_parents.append((uid, str_map[p]))
            except ValueError: continue

            if len(batch_nodes) >= BATCH_SIZE:
                c.executemany("INSERT INTO nodes VALUES (?,?,?,?,?,?)", batch_nodes)
                c.executemany("INSERT INTO node_parents VALUES (?,?)", batch_parents)
                batch_nodes.clear()
                batch_parents.clear()
        
        # Flush remaining
        if batch_nodes: c.executemany("INSERT INTO nodes VALUES (?,?,?,?,?,?)", batch_nodes)
        if batch_parents: c.executemany("INSERT INTO node_parents VALUES (?,?)", batch_parents)
        
        # B. Predicates
        batch_preds = []
        
        for fname, leads_uri in [("classes_predicates.tsv", 0), ("classes_predicates_no_literals.tsv", 1)]:
            for row in valid_rows(os.path.join(path, fname), 5):
                if row[4] == "0": continue
                cls, start_str, dir_str, pred, pop = row[0], row[1], row[2], row[3], row[4]
                
                if cls not in str_map or pred not in str_map: continue
                
                s_enum = 1 if "direct" in start_str.lower() else 2
                d_enum = 1 if "outgoing" in dir_str.lower() else 2
                
                batch_preds.append((str_map[cls], str_map[pred], d_enum, s_enum, int(pop), leads_uri))
                
                if len(batch_preds) >= BATCH_SIZE:
                    c.executemany("INSERT INTO node_predicates VALUES (?,?,?,?,?,?)", batch_preds)
                    batch_preds.clear()

        if batch_preds: c.executemany("INSERT INTO node_predicates VALUES (?,?,?,?,?,?)", batch_preds)
        
        # C. URIL Mappings (Identity)
        print("      Generating Mappings...")
        c.execute("INSERT INTO uril_mappings (uri_id, uril_id) SELECT id, id FROM nodes")

        self.conn.commit()

        # --- PHASE 5: INDICES ---
        print("[5/5] Building Final Indices...")
        # Create indices AFTER inserting. This is O(N) instead of O(N log N) during insert.
        c.execute("CREATE INDEX idx_node_parents_id ON node_parents(node_id)")
        c.execute("CREATE INDEX idx_node_predicates_lookup ON node_predicates(node_id, direction, start_point)")
        self.conn.commit()
        
    # --- URI / URIL MAPPING LOGIC ---
    
    def register_uri_to_uril_mapping(self, uri: str):
        self.uri_to_uril_map[uri] = uri
        self.uril_to_uri_map[uri] = uri

    def uri_to_uril(self, uri: str):
        if not is_uri(uri): return uri
        
        # Strip brackets for lookup
        clean_uri = uri[1:-1] if uri.startswith("<") else uri
        
        if clean_uri in self.uri_to_uril_map:
            res_val = self.uri_to_uril_map[clean_uri]
            if uri.startswith("<"): return f"<{res_val}>"
            return res_val
        
        # Miss logic
        self.register_uri_to_uril_mapping(clean_uri)
        uril = self.uri_to_uril_map[clean_uri]
        if uri.startswith("<"): return f"<{uril}>"
        return uril

    def uris_to_urils(self, uris: list):
        return [self.uri_to_uril(u) for u in uris]
    
    def uril_to_uri(self, uril: str):
        if not is_uri(uril): return uril
        clean_uril = uril[1:-1] if uril.startswith("<") else uril
        
        if clean_uril in self.uril_to_uri_map:
            res_val = self.uril_to_uri_map[clean_uril]
            if uril.startswith("<"): return f"<{res_val}>"
            return res_val
            
        return uril

    def urils_to_uris(self, urils: list):
        return [self.uril_to_uri(u) for u in urils]

    def triples_with_urils_to_triples_with_uris(self, triples: str):
        return re.sub(r'<([^>]*)>', lambda m: f"<{self.uril_to_uri(m.group(1))}>", triples)

    def triples_with_uris_to_triples_with_urils(self, triples: str):
        return re.sub(r'<([^>]*)>', lambda m: f"<{self.uri_to_uril(m.group(1))}>", triples)
    
    def get_label_from_uri(self, uri: str) -> Optional[str]:
        # Fast DB lookup for label if exists
        comp = self.get_kg_component(uri)
        if comp and comp.label:
            return comp.label
            
        # Fallback to string manipulation if no label in DB
        uril = self.uri_to_uril(uri)
        additional = len(uril) - len(uri)
        if additional > 0:
            return uril[-additional:].replace("_", " ")
        label = re.split(r'[/#?]', uri)[-1]
        return label.replace("_", " ")
    
    # --------------------------------------
    # ----- SPARQL Execution Utilities -----
    # --------------------------------------
    
    def are_triples_valid(self, triples: str, prefixes: str = "") -> bool:
        start_time = time.time()
        # triples = self.triples_with_urils_to_triples_with_uris(triples)
        
        query = f"""
        {prefixes}
        ASK WHERE {{ 
            {triples}
        }}
        """
        log(f"are_triples_valid: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
        
        try:
            query_result = execute_sparql_query(query, self.endpoint)
            result = query_result.convert()
            get_kgaqa_tracker()._are_triples_valid_time += time.time() - start_time
            get_kgaqa_tracker()._are_triples_valid_calls += 1
            return result['boolean']
        except Exception as e:
            log(f"Error are_triples_valid: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            # raise ValueError("Error are_triples_valid")
            get_kgaqa_tracker()._are_triples_valid_time += time.time() - start_time
            get_kgaqa_tracker()._are_triples_valid_calls += 1
            return False
        
    def has_no_value_connection(self, triples: str):    
        start_time = time.time()
          
        query = f"""
        SELECT * WHERE {{ 
            {triples}
        }} LIMIT 1
        """
        log(f"has_no_value_connection: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.DEBUG)
        
        try:
            query_result = execute_sparql_query(query, self.endpoint)
            results = query_result.convert()
        except Exception as e:
            log(f"Error has_no_value_connection: {e}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            log(f"Query: {query}", LogComponent.PATH_EXTRACTOR, LogLevel.CRITICAL)
            get_kgaqa_tracker()._has_no_value_connection_time += time.time() - start_time
            get_kgaqa_tracker()._has_no_value_connection_calls += 1
            return False
        
        # Filter any queries without results
        if len(results["results"]["bindings"]) == 0:
            get_kgaqa_tracker()._has_no_value_connection_time += time.time() - start_time
            get_kgaqa_tracker()._has_no_value_connection_calls += 1
            return False
        
        # Filter any results that contain literals
        keep = True
        for result in results["results"]["bindings"]:   
            for var in result:
                if (result[var]["type"]) == 'literal':
                    keep = False
                    break
            if keep == False:
                break
            
        get_kgaqa_tracker()._has_no_value_connection_time += time.time() - start_time
        get_kgaqa_tracker()._has_no_value_connection_calls += 1
        
        return keep
    
    def get_values_for_triples(self, triples_string:str, k: int = 3, prefixes: str = "") -> List[Dict[str, str]]:        
        query = prefixes + """
        SELECT * WHERE {
        """ + triples_string + """
        } 
        LIMIT """ + str(k)
        # print(query)
        try:
            results = execute_sparql_query(query, self.endpoint)
            results = results.convert()
            # Standard SPARQL JSON for SELECT looks like: { "results": { "bindings": [...] } }
            if 'results' in results:
                bindings = results['results']['bindings']
                simplified_list = []
                
                for row in bindings:
                    clean_item = {}
                    for key, value_obj in row.items():
                        # We strip the metadata (type, datatype) and just keep the value
                        clean_item[key] = value_obj['value']
                    if clean_item == {}:
                        continue
                    simplified_list.append(clean_item)
                # print(simplified_list)
                return simplified_list

        except Exception as e:
            print(f"Error processing SPARQL results: {str(e)}")
            raise e 
        return []
    
    def get_object_for_node_predicate_info(self, node_uri: str, predicate_info: PredicateInfo, k: int = 3) -> List[str]:
        direction = predicate_info.get_direction_word()
        predicate = predicate_info.uri
        triples = ""
        if direction == "to":
            triples = f"<{node_uri}> <{predicate}> ?o ."
        elif direction == "from":
            triples = f"?o <{predicate}> <{node_uri}> ."
        else:
            return []
        
        results = self.get_values_for_triples(triples, k)
        objects = []
        for res in results:
            if 'o' in res:
                objects.append(res['o'])
        
        return objects
    
    def shorten_uri(self, uri: str) -> str:
        # Do not shorten URIs which would be syntactically invalid
        if "," in uri or " " in uri or "'" in uri or '"' in uri:
            return uri
        prefix_map = self.prefixes_map
        shortest = uri
        for prefix, full_uri in prefix_map.items():
            if uri.startswith(full_uri):
                new = prefix + ":" + uri[len(full_uri):]
                if len(new) < len(shortest):
                    shortest = new
        return shortest
    
    def expand_uri(self, short_uri: str) -> str:
        prefix_map = self.prefixes_map
        if ":" not in short_uri:
            return short_uri
        prefix, local_part = short_uri.split(":", 1)
        if prefix in prefix_map:
            return prefix_map[prefix] + local_part
        return short_uri