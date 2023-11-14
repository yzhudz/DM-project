import sys
from typing import Dict, List
from sketch import Sketch
import random
import mmh3
import utils as utils
import numpy as np
import ground_truth
from evaluation import metrics
import pandas as pd
import math


def l2_norm(elements: List[int], a_value: List[int]):
    value = 0
    for index, element in enumerate(elements):
        if a_value[index] != 0:
            value += (element / a_value[index]) ** 2
    return math.sqrt(value)


class OurUSS(Sketch):

    def __init__(self, params: Dict[str, int]):

        # self.file_path = file_path
        super().__init__(params)
        # self.stream_item_set = []
        self.hash_function_num = params["hash_function_nums"]
        self.bucket_num = params["bucket_num"]
        self.buckets = [[{"key": -1, "value": 0} for _ in range(self.bucket_num)] for _ in
                        range(self.hash_function_num)]
        self.value_count = params["value_count"]

        self.normalization = params['normalization']
        self.a_value = [0 if self.normalization ==
                             True else 1] * self.value_count
        # self.exact_counter = defaultdict(int)
        # self.space_saving_count = defaultdict(int)
        # self.amount_of_words = 0

    def insert(self, key: int, value: List[int]):
        empty_bucket = -1
        empty_pos_in_bucket = -1
        min_value = float('inf')
        min_bucket = -1
        min_pos_in_bucket = -1

        for i in range(self.hash_function_num):
            bucket_i_pos = mmh3.hash(key.to_bytes(
                length=8, byteorder='big'), seed=i) % self.bucket_num
            element = self.buckets[i][bucket_i_pos]
            # Element key is recorded in one of the buckets, simply increase the value.
            if element["key"] == key:
                for j in range(self.value_count):
                    element["value"][j] += value[j]
                return
            # An empty slot is found, first record the position.
            if element["key"] == -1:
                if empty_bucket < 0:
                    empty_bucket = i
                    empty_pos_in_bucket = bucket_i_pos

            # Get the min_value_pos
            else:
                l2 = l2_norm(element["value"], self.a_value)
                if l2 < min_value:
                    min_value = l2
                    min_bucket = i
                    min_pos_in_bucket = bucket_i_pos

        # After the loop, checking if there is any empty bucket and place the element in the empty position.
        if empty_bucket >= 0:
            element = self.buckets[empty_bucket][empty_pos_in_bucket]
            element["key"] = key
            element["value"] = value.copy()
            return

        # If there is no empty, compete the incoming element with the element of minimum.
        element = self.buckets[min_bucket][min_pos_in_bucket]
        # Compute the drop probability
        drop_probability = 1 / (min_value + 1)
        rand_number = np.random.uniform(low=0, high=1)
        # update the key and value according to probability
        if rand_number < drop_probability:
            element["key"] = key
            element["value"] = value

    def all_query(self) -> Dict[int, List[int]]:
        result = {}
        for bucket in self.buckets:
            for element in bucket:
                if element["key"] == -1:
                    continue
                if element["key"] in result:
                    for i in range(self.value_count):
                        result[element["key"]][i] += element["value"][i]
                else:
                    result[element["key"]] = element["value"]
        return result


if __name__ == "__main__":
    USS = OurUSS({"hash_function_nums": 2, "value_count": 5,
                  "bucket_num": 1000, "normalization": False})
    groundTruth = ground_truth.GroundTruth(
        {"value_count": 5, "normalization": True})
    with open("./synthetic_dataset/synthetic_dataset.txt") as f:
        line = f.readline()
        while line:
            row = line.split()
            key = row[0]
            value = row[1:]
            value = [int(x) for x in value]
            USS.insert(int(key), value)
            groundTruth.insert(int(key), value)
            line = f.readline()
        f.close()
    result = USS.all_query()
    result2, truth_a_value = groundTruth.all_query()
    print(result[1])
    print(result2[1])
    print(pd.DataFrame(result).T.reset_index())
    print(str(pd.DataFrame(result).T.reset_index().columns))
    print('f1 score: ', metrics(result, result2, truth_a_value))
