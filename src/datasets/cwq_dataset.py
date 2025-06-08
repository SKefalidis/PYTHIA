import json
from src.datasets.dataset import Dataset
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs


FREEBASE_PREFIXES = """
    PREFIX uom: <http://www.opengis.net/def/uom/OGC/1.0/>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX ns: <http://rdf.freebase.com/ns/>
"""


class CwqDataset(Dataset):
    def __init__(self, dataset):
        super().__init__("CWQ")
        
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
        return entry['question']
    
    def get_query(self, entry):
        return entry['sparql']
    
    @staticmethod
    def get_prefixes():
        return FREEBASE_PREFIXES
    
    def get_knowledge_graph(self):
        return KnowledgeGraphs.FREEBASE
    