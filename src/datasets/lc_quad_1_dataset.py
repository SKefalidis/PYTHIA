import json
from src.engine.gost_requests import extract_uris
from src.datasets.dataset import Dataset
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from tqdm import tqdm


class LcQuad1Dataset(Dataset):
    def __init__(self, dataset):
        super().__init__("LC-QuAD")
        
        self.dataset = dataset

    @classmethod 
    def from_files(cls, file_path: str):
        # Load the dataset
        data_file = open(file_path)
        dataset = json.load(data_file)
        data_file.close()
                
        return cls(dataset)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx >= len(self.dataset):
            raise IndexError("Index out of range")
        return self.dataset[idx]
    
    def get_question(self, entry):
        return entry['corrected_question']
    
    def get_query(self, entry):
        return entry['sparql_query']
    
    def get_prefixes(self):
        return ""
    
    def get_knowledge_graph(self):
        return KnowledgeGraphs.DBPEDIA
    