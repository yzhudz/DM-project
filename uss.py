from typing import Dict, List
from sketch import Sketch
import mmh3
import utils as utils
import numpy as np
import ground_truth
from evaluation import metrics
import pandas as pd


class OurUSS(Sketch):

    def __init__(self, params: Dict[str, int]):
        super().__init__(params)
        self.hash_function_num = params["hash_function_nums"]
        self.bucket_num = params["bucket_num"]
        self.buckets = [[{"key": -1, "value": 0} for _ in range(self.bucket_num)] for _ in
                        range(self.hash_function_num)]

    def insert(self, key: int, value: int):
        min_value = float('inf')
        min_bucket = -1
        min_pos_in_bucket = -1

        for i in range(self.hash_function_num):
            bucket_i_pos = mmh3.hash(key.to_bytes(
                length=8, byteorder='big'), seed=i) % self.bucket_num
            element = self.buckets[i][bucket_i_pos]
            # Element key is recorded in one of the buckets, simply increase the value.
            if element["key"] == key:
                element["value"] += value
                return
            # An empty slot is found, insert the value.
            if element["key"] == -1:
                element = self.buckets[i][bucket_i_pos]
                element["key"] = key
                element["value"] = value
                return

            # Get the min_value_pos
            current_value = element['value']
            if current_value < min_value:
                min_bucket = i
                min_pos_in_bucket = bucket_i_pos
                min_value = current_value
        if value == 0:
            return

        # min value position
        element = self.buckets[min_bucket][min_pos_in_bucket]
        # Compute the drop probability
        drop_probability = value / (min_value + value)
        rand_number = np.random.uniform(low=0, high=1)
        # update the key and value according to probability
        # replace
        if rand_number < drop_probability:
            element["key"] = key
            element["value"] = value
        # not replace, add the value of min
        else:
            self.buckets[min_bucket][min_pos_in_bucket]['value'] += value

    def all_query(self) -> Dict[int, List[int]]:
        result = {}
        for bucket in self.buckets:
            for element in bucket:
                if element["key"] == -1:
                    continue
                result[element["key"]] = element["value"]
        return result


if __name__ == "__main__":
    value_count = 5
    USS = [OurUSS({"hash_function_nums": 2,
                  "bucket_num": 1000}) for _ in range(value_count)]
    groundTruth = ground_truth.GroundTruth(
        {"value_count": 5, "normalization": False})
    with open("./synthetic_dataset/synthetic_dataset.txt") as f:
        line = f.readline()
        while line:
            row = line.split()
            key = row[0]
            value = row[1:]
            value = [int(x) for x in value]
            for i in range(value_count):
                USS[i].insert(int(key), value[i])
            groundTruth.insert(int(key), value)
            line = f.readline()
        f.close()

    result = {}
    for i in range(value_count):
        single_result = USS[i].all_query()
        for j, k in single_result.items():
            if j not in result:
                result[j] = [0] * value_count
            result[j][i] = single_result[j]

    result2, truth_a_value = groundTruth.all_query()
    print(result[1])
    print(result2[1])
    print(pd.DataFrame(result).T.reset_index())
    print(str(pd.DataFrame(result).T.reset_index().columns))
    print('f1 score: ', metrics(result, result2, truth_a_value))
