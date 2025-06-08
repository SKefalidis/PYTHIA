import json
from src.engine.gost_requests import extract_uris
from src.datasets.dataset import Dataset
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from tqdm import tqdm



DBPEDIA_PREFIXES = """
    PREFIX bd: <http://www.bigdata.com/rdf#>
    PREFIX cc: <http://creativecommons.org/ns#>
    PREFIX geo: <http://www.opengis.net/ont/geosparql#>
    PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX res: <http://dbpedia.org/resource/> 
    PREFIX dbp: <http://dbpedia.org/property/> 
    PREFIX dbpedia2: <http://dbpedia.org/property/>
    PREFIX dct: <http://purl.org/dc/terms/> 
    PREFIX dbc: <http://dbpedia.org/resource/Category:>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
    PREFIX onto: <http://dbpedia.org/ontology/>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX yago: <http://yago-knowledge.org/resource/>
"""


class Qald9Dataset(Dataset):
    def __init__(self, dataset):
        super().__init__("QALD-9")
        
        self.dataset = dataset

    @classmethod 
    def from_files(cls, file_path: str):
        # Load the dataset
        data_file = open(file_path)
        dataset = json.load(data_file)
        data_file.close()
        
        entries = []
        for entry in dataset['questions']:
            new_entry = {}
            for question in entry['question']:
                if question['language'] == 'en':
                    new_entry['question'] = question['string']
            new_entry['query'] = entry['query']['sparql']
            new_entry['answers'] = []
            for answer in entry['answers']:
                if 'boolean' in answer:
                    new_entry['answers'] = answer['boolean']
                else:
                    for result in answer['results']['bindings']:
                        # print(result)
                        for key in result.keys():
                            # print(key)
                            new_entry['answers'].append(result[key]['value'])
            entries.append(new_entry)
                
        return cls(entries)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx >= len(self.dataset):
            raise IndexError("Index out of range")
        return self.dataset[idx]
    
    def get_question(self, entry):
        return entry['question']
    
    def get_query(self, entry):
        return entry['query']
    
    def get_prefixes(self):
        return DBPEDIA_PREFIXES
    
    def get_knowledge_graph(self):
        return KnowledgeGraphs.DBPEDIA10