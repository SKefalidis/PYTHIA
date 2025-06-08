from abc import ABC, abstractmethod

class Evaluatable(ABC):
    """
    An interface for objects that can be evaluated.
    """
    
    @abstractmethod
    def get_name(self):
        pass
    
    @abstractmethod
    def get_resource(self):
        pass

    @abstractmethod
    def predict(self, question: str, logging: bool = False):
        pass
    
    def __str__(self):
        return self.get_name()