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

    def to_dict(self):
        return {
            "QUESTIONS": self.QUESTIONS,
            "TIME": self.TIME,
            "TIME_PER_QUESTION": self.TIME / self.QUESTIONS if self.QUESTIONS > 0 else 0,
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