import argparse
from torch.utils.data import Dataset as TorchDataset
from src.knowledge_graphs.knowledge_graphs import KnowledgeGraphs
from abc import abstractmethod



class Dataset(TorchDataset):
    def __init__(self, name):
        self._name = name
        
    @classmethod 
    def from_files(cls, file_path: str):
        """
        Factory method to create a Dataset instance from a file path.
        This method should be implemented by subclasses to handle specific file formats.
        """
        raise NotImplementedError("from_files method not implemented")
        
    def get_name(self):
        return self._name
    
    @abstractmethod
    def get_question(self, entry):
        raise NotImplementedError("get_question method not implemented")
    
    @abstractmethod
    def get_query(self, entry):
        raise NotImplementedError("get_query method not implemented")
    
    @abstractmethod
    def get_prefixes(self):
        raise NotImplementedError("get_prefixes method not implemented")
    
    @abstractmethod
    def get_knowledge_graph(self) -> KnowledgeGraphs:
        raise NotImplementedError("get_knowledge_graph method not implemented")
    
    def __str__(self):
        return self.get_name()


class DatasetFactory:
    
    @staticmethod
    def list_datasets() -> list[str]:
        return ["qald-9", "qald-10", "lc-quad-1", "beastiary", "bestiary", "webqsp", "cwq", "graphq", "grailqa", "spinach"]
    
    @staticmethod
    def create_dataset(dataset: str, file_path: str) -> Dataset:
        if dataset.lower() == "qald-9":
            from src.datasets.qald9_dataset import Qald9Dataset
            return Qald9Dataset.from_files(file_path)
        elif dataset.lower() == "qald-10":
            from src.datasets.qald10_dataset import Qald10Dataset
            return Qald10Dataset.from_files(file_path)
        elif dataset.lower() == "lc-quad-1":
            from src.datasets.lc_quad_1_dataset import LcQuad1Dataset
            return LcQuad1Dataset.from_files(file_path)
        elif dataset.lower() == "beastiary" or dataset.lower() == "bestiary":
            from src.datasets.beastiary_dataset import BeastiaryDataset
            return BeastiaryDataset.from_files(file_path)
        elif dataset.lower() == "webqsp":
            from src.datasets.webqsp_dataset import WebQSPDataset
            return WebQSPDataset.from_files(file_path)
        elif dataset.lower() == "cwq":
            from src.datasets.cwq_dataset import CwqDataset
            return CwqDataset.from_files(file_path)
        elif dataset.lower() == "graphq":
            from src.datasets.graphquestions_dataset import GraphQuestionsDataset
            return GraphQuestionsDataset.from_files(file_path)
        elif dataset.lower() == "grailqa":
            from src.datasets.grailqa_dataset import GrailQaDataset
            return GrailQaDataset.from_files(file_path)
        elif dataset.lower() == "spinach":
            from src.datasets.spinach_dataset import SpinachDataset
            return SpinachDataset.from_files(file_path)
        else:
            raise ValueError(f"Unknown dataset name: {dataset}")
        
    @staticmethod
    def create_from_args(args) -> Dataset:
        if not hasattr(args, "dataset") or not hasattr(args, "dataset_path"):
            raise ValueError("Arguments must include dataset and dataset_path")
        return DatasetFactory.create_dataset(args.dataset, args.dataset_path)
    
    @staticmethod
    def fill_parse_args(parser: argparse.ArgumentParser, argument_group: argparse._ArgumentGroup = None) -> argparse._ArgumentGroup:
        """Adds dataset-related arguments to the provided argument parser or argument group.
            Added arguments:
            --dataset: The dataset to use
            --dataset_path: The path to the dataset file
        """
        datasets = DatasetFactory.list_datasets()
        if argument_group is not None:
            dataset_group = argument_group
        else:
            dataset_group = parser.add_argument_group("Dataset Settings")
        dataset_group.add_argument("--dataset", type=str, required=True, choices=datasets, default="qald-9",
                                   help="The dataset to use")
        dataset_group.add_argument("--dataset_path", type=str, required=True,
                                   help="The path to the dataset file")
        return dataset_group