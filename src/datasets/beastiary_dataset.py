import json
from src.datasets.dataset import Dataset
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs


BEASTIARY_PREFIXES = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

class BeastiaryDataset(Dataset):
    def __init__(self, dataset):
        super().__init__("Beastiary")
        
        self.dataset = dataset

    @classmethod 
    def from_files(cls, file_path: str):
        # Load the dataset
        data_file = open(file_path)
        dataset = json.load(data_file)
        data_file.close()
        
        entries = []
        for entry in dataset['questions']:
            # print(entry)
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
                        if 'boolean' in result:
                            new_entry['answers'] = result['boolean']
                            continue
                        for key in result.keys():
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
        return BEASTIARY_PREFIXES
    
    def get_knowledge_graph(self):
        return KnowledgeGraphs.BEASTIARY_KG
    