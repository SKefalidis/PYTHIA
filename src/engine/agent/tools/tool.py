from abc import ABC, abstractmethod
from typing import Any

class Tool(ABC):
    
    def __init__(self):
        pass
    
    @classmethod
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def schema(self) -> dict:
        raise NotImplementedError
    
    @abstractmethod
    def function(self, *args, **kwargs) -> Any:
        raise NotImplementedError