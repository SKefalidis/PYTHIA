import re

from abc import abstractmethod
from src.evaluation.evaluatable import Evaluatable


class ClassIdentifier(Evaluatable):

    def __init__(self, top_k: int = 0):
        self.top_k = top_k # FIXME
        self.retriever = None

    def extract(self, text: str):
        return re.findall(r'http[^\s]+', text)

    def identify(self, question: str, top_k: int = None, threshold: float = 0.0, debug: bool = False, return_labels: bool = False, logging: bool = False):
        if top_k is not None:
            self.top_k = top_k

        self.retriever.similarity_top_k = self.top_k

        try:
            response = self.retriever.retrieve(question)           
        except Exception as e:
            print(question)
            print(e)
        pruned_nodes = [node for node in response if node.score > threshold]
        # pruned_nodes = self.reranker._postprocess_nodes(response, QueryBundle(question))
        classes = [node.get_text().split(" - ")[0].strip() for node in pruned_nodes]

        if debug:
            print("Question:", question)
            print("Response:")
            for node in pruned_nodes:
                print(node)
                print(node.score)

        if return_labels == False:
            if logging == False:
                return classes
            else:
                return classes, None
        else:
            labels = [node.get_text().split(" - ")[1].strip() for node in pruned_nodes]
            return list(zip(classes, labels))

    def predict(self, question: str, logging: bool = False):
        return self.identify(question, logging=logging)
    
    @staticmethod 
    @abstractmethod
    def get_name():
        pass
    
    @abstractmethod
    def get_resource(self):
        pass