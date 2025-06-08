class Metrics:
    def __init__(self):
        self.QUESTIONS = 0
        self.TIME = 0
        self.SPARQL_CALLS = 0
        self.SPARQL_TIME = 0.0
        self.LLM_CALLS = 0
        self.LLM_TIME = 0.0
        self.LLM_INPUTS = 0
        self.LLM_OUTPUTS = 0
        
    def load_from_file(self, file_path):
        import json
        with open(file_path, "r") as infile:
            data = json.load(infile)
            self.QUESTIONS = data.get("QUESTIONS", 0)
            self.TIME = data.get("TIME", 0.0)
            self.SPARQL_CALLS = data.get("SPARQL_CALLS", 0)
            self.SPARQL_TIME = data.get("SPARQL_TIME", 0.0)
            self.LLM_CALLS = data.get("LLM_CALLS", 0)
            self.LLM_TIME = data.get("LLM_TIME", 0.0)
            self.LLM_INPUTS = data.get("LLM_INPUTS", 0)
            self.LLM_OUTPUTS = data.get("LLM_OUTPUTS", 0)

    def to_dict(self):
        return {
            "QUESTIONS": self.QUESTIONS,
            "TIME": self.TIME,
            "TIME PER QUESTION": self.TIME / self.QUESTIONS if self.QUESTIONS > 0 else 0,
            "SPARQL_CALLS": self.SPARQL_CALLS,
            "SPARQL_TIME": self.SPARQL_TIME,
            "LLM_CALLS": self.LLM_CALLS,
            "LLM_TIME": self.LLM_TIME,
            "LLM_INPUTS": self.LLM_INPUTS,
            "LLM_OUTPUTS": self.LLM_OUTPUTS
        }
    
    def add_metrics(self, other):
        self.QUESTIONS += 1
        self.TIME += other.TIME
        self.SPARQL_CALLS += other.SPARQL_CALLS
        self.SPARQL_TIME += other.SPARQL_TIME
        self.LLM_CALLS += other.LLM_CALLS
        self.LLM_TIME += other.LLM_TIME
        self.LLM_INPUTS += other.LLM_INPUTS
        self.LLM_OUTPUTS += other.LLM_OUTPUTS

GLOBAL_METRICS = None
def get_global_metrics():
    global GLOBAL_METRICS
    if GLOBAL_METRICS is None:
        GLOBAL_METRICS = Metrics()
    return GLOBAL_METRICS

CURRENT_METRICS = None
def get_current_metrics(new=False):
    global CURRENT_METRICS
    if CURRENT_METRICS is None or new==True:
        CURRENT_METRICS = Metrics()
    return CURRENT_METRICS

USER = "user"
def set_user(user):
    global USER
    USER = user
def get_user():
    global USER
    return USER

PASSWORD = "user"
def set_password(password):
    print("Setting password to: ", password)
    global PASSWORD
    PASSWORD = password
def get_password():
    global PASSWORD
    return PASSWORD

import time

TIMER = 0.0
def set_timer_to_current():
    global TIMER
    TIMER = time.time()
def get_timer():
    global TIMER
    return TIMER

FREEBASE_PREFIXES = [
    ("uom", "http://www.opengis.net/def/uom/OGC/1.0/"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
    ("ns", "http://rdf.freebase.com/ns/"),
]
FREEBASE_LABEL_PREDICATE = "http://rdf.freebase.com/ns/type.object.name"

WIKIDATA_PREFIXES = [
    ("bd", "http://www.bigdata.com/rdf#"),
    ("cc", "http://creativecommons.org/ns#"),
    ("dct", "http://purl.org/dc/terms/"),
    ("geo", "http://www.opengis.net/ont/geosparql#"),
    ("ontolex", "http://www.w3.org/ns/lemon/ontolex#"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("p", "http://www.wikidata.org/prop/"),
    ("pq", "http://www.wikidata.org/prop/qualifier/"),
    ("pqn", "http://www.wikidata.org/prop/qualifier/value-normalized/"),
    ("pqv", "http://www.wikidata.org/prop/qualifier/value/"),
    ("pr", "http://www.wikidata.org/prop/reference/"),
    ("prn", "http://www.wikidata.org/prop/reference/value-normalized/"),
    ("prov", "http://www.w3.org/ns/prov#"),
    ("prv", "http://www.wikidata.org/prop/reference/value/"),
    ("ps", "http://www.wikidata.org/prop/statement/"),
    ("psn", "http://www.wikidata.org/prop/statement/value-normalized/"),
    ("psv", "http://www.wikidata.org/prop/statement/value/"),
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("schema", "http://schema.org/"),
    ("skos", "http://www.w3.org/2004/02/skos/core#"),
    ("wd", "http://www.wikidata.org/entity/"),
    ("wdata", "http://www.wikidata.org/wiki/Special:EntityData/"),
    ("wdno", "http://www.wikidata.org/prop/novalue/"),
    ("wdsubgraph", "https://query.wikidata.org/subgraph/"),
    ("wdref", "http://www.wikidata.org/reference/"),
    ("wds", "http://www.wikidata.org/entity/statement/"),
    ("wdt", "http://www.wikidata.org/prop/direct/"),
    ("wdtn", "http://www.wikidata.org/prop/direct-normalized/"),
    ("wdv", "http://www.wikidata.org/value/"),
    ("wikibase", "http://wikiba.se/ontology#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
]
WIKIDATA_LABEL_PREDICATE = "http://www.w3.org/2000/01/rdf-schema#label"

DBPEDIA_PREFIXES = [
    ("bd", "http://www.bigdata.com/rdf#"),
    ("cc", "http://creativecommons.org/ns#"),
    ("geo", "http://www.opengis.net/ont/geosparql#"),
    ("ontolex", "http://www.w3.org/ns/lemon/ontolex#"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("prov", "http://www.w3.org/ns/prov#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
    ("res", "http://dbpedia.org/resource/"),
    ("dbp", "http://dbpedia.org/property/"),
    ("dbpedia2", "http://dbpedia.org/property/"),
    ("dct", "http://purl.org/dc/terms/"),
    ("dbc", "http://dbpedia.org/resource/Category:"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("onto", "http://dbpedia.org/ontology/"),
    ("dbo", "http://dbpedia.org/ontology/"),
    ("dbr", "http://dbpedia.org/resource/"),
    ("yago", "http://yago-knowledge.org/resource/"),
]
DBPEDIA_LABEL_PREDICATE = "http://www.w3.org/2000/01/rdf-schema#label"

BESTIARY_PREFIXES = [
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("bst", "http://www.semanticweb.org/annab/ontologies/2022/3/ontology#"),
]
BESTIARY_LABEL_PREDICATE = ""

YAGO2GEO_PREFIXES = [
    ("geo", "http://www.opengis.net/ont/geosparql#"),
    ("geof", "http://www.opengis.net/def/function/geosparql/"),
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
    ("yago", "http://yago-knowledge.org/resource/"),
    ("y2geor", "http://kr.di.uoa.gr/yago2geo/resource/"),
    ("y2geoo", "http://kr.di.uoa.gr/yago2geo/ontology/"),
    ("strdf", "http://strdf.di.uoa.gr/ontology#"),
    ("uom", "http://www.opengis.net/def/uom/OGC/1.0/"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
]
YAGO2GEO_LABEL_PREDICATE = [
    "http://kr.di.uoa.gr/yago2geo/ontology/hasOSM_Name",
    "http://kr.di.uoa.gr/yago2geo/ontology/hasGADM_Name",
    "http://kr.di.uoa.gr/yago2geo/ontology/hasOS_Name",
    "http://kr.di.uoa.gr/yago2geo/ontology/hasGAG_Name",
    "http://kr.di.uoa.gr/yago2geo/ontology/hasGAG_Name",
    "http://kr.di.uoa.gr/yago2geo/ontology/hasOSNI_Name",
    "http://kr.di.uoa.gr/yago2geo/ontology/hasOSI_Name"
]

LABEL_PREDICATE = None
PREFIXES_LIST = None
PREFIXES = None

def get_prefixes(prefixes):
    return '\n'.join(['PREFIX %s: <%s>' % (prefix[0], prefix[1]) for prefix in prefixes])
