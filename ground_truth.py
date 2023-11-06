from abc import ABC, abstractmethod
from sketch import Sketch
from typing import Dict, Tuple, Sequence

"""
  This is the most naive algorithm for calculating the exact sum of each key. 
  It guarantees accurate results, serving as a reference for comparison with 
  various sketch algorithms in experiments and control groups.
"""
class GroundTruth(Sketch):

    def __init__(self, params: Dict[str, int]):
        super().__init__(params)
        self.params = params
        self.result = {}
        self.value_count = params["value_count"]
        self.normalization = params['normalization']
        self.a_value = [0 if self.normalization == True else 1] * self.value_count


    def insert(self, key, value):

        if self.normalization:
            for i in range(len(self.a_value)):
                self.a_value[i] += value[i]

        if key in self.result:
            for i in range(self.value_count):
                self.result[key][i] += value[i]
        else:
            self.result[key] = value

    def all_query(self):
        return self.result, self.a_value
