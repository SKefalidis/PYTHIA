import yaml
import json
from src.utils_gost import extract_uris
from src.dataloaders.dataset import Dataset
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from tqdm import tqdm


CK26_PREFIXES = """
    PREFIX bd: <http://www.bigdata.com/rdf#>
    PREFIX cc: <http://creativecommons.org/ns#>
    PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX dct: <http://purl.org/dc/terms/> 
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
    PREFIX pvoc: <http://ld.company.org/prod-vocab/>
    PREFIX pins: <http://ld.company.org/prod-instances/>
"""


class T2SCK26Dataset(Dataset):
    def __init__(self, dataset):
        super().__init__("T2S-CK26")
        self.dataset = dataset

    @classmethod 
    def from_files(cls, file_path: str):
        # Load the dataset
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            data_file = open(file_path)
            dataset = yaml.safe_load(data_file)
            data_file.close()
            
            entries = []
            for entry in dataset['questions']:
                new_entry = {}
                new_entry['question'] = entry['question']['en']
                new_entry['query'] = entry['query']['sparql']
                entries.append(new_entry)
        else:
            data_file = open(file_path)
            dataset = json.load(data_file)
            data_file.close()
            
            entries = []
            for entry in dataset:
                new_entry = {}
                new_entry['question'] = entry['question']
                new_entry['query'] = entry['query']
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
    
    def get_prefixes(self):# -> Any:
        return CK26_PREFIXES
    
    def get_knowledge_graph(self):
        return KnowledgeGraphs.CORPORATE_KG