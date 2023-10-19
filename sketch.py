from abc import ABC, abstractmethod
from typing import Dict, Tuple, Sequence


class Sketch(ABC):

    def __init__(self, params: Dict[str, int]):
        self.params = params
        pass

    @abstractmethod
    def insert(self, key, value):
        pass

    @abstractmethod
    def all_query(self):
        pass
