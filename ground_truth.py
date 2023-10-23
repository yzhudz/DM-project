from abc import ABC, abstractmethod
from sketch import Sketch
from typing import Dict, Tuple, Sequence


class GroundTruth(Sketch):

    def __init__(self, params: Dict[str, int]):
        super().__init__(params)
        self.params = params
        self.result = {}
        self.value_count = params["value_count"]

    def insert(self, key, value):
        if key in self.result:
            for i in range(self.value_count):
                self.result[key][i] += value[i]
        else:
            self.result[key] = value

    def all_query(self):
        return self.result
