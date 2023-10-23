import sys
from typing import Dict, List
from sketch import Sketch
import mmh3
import utils as utils
import numpy as np
import ground_truth


class HyperUSS(Sketch):

    def __init__(self, params: Dict[str, int]):
        super().__init__(params)
        # d hash functions (bucket arrays)
        self.hash_function_num = params["hash_function_nums"]
        # w buckets in each array
        self.bucket_num = params["bucket_num"]
        # value count
        self.value_count = params["value_count"]
        # d * w buckets
        self.buckets = [[{"key": -1, "value": []} for _ in range(self.bucket_num)] for _ in
                        range(self.hash_function_num)]
        # Normalize array, which refers to "A" in the paper
        self.normalize_values = [0 for _ in range(self.value_count)]

    def insert(self, key: int, value: List[int]):
        empty_bucket = -1
        empty_pos_in_bucket = -1
        min_l2_norm = sys.maxsize
        min_l2_norm_bucket = -1
        min_l2_pos_in_bucket = -1

        # First add the normalize array
        for i in range(self.value_count):
            self.normalize_values[i] += value[i]

        for i in range(self.hash_function_num):
            bucket_i_pos = mmh3.hash(key.to_bytes(length=8, byteorder='big'), seed=i) % self.bucket_num
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
            # compute l2 norm of the bucket, and record the minimum one
            else:
                l2 = utils.normalized_l2_norm(element["value"], self.normalize_values)
                if l2 < min_l2_norm:
                    min_l2_norm = l2
                    min_l2_norm_bucket = i
                    min_l2_pos_in_bucket = bucket_i_pos

        # After the loop, checking if there is any empty bucket and place the element in the empty position.
        if empty_bucket >= 0:
            element = self.buckets[empty_bucket][empty_pos_in_bucket]
            element["key"] = key
            element["value"] = value.copy()
            return

        # If there is no empty, compete the incoming element with the element of minimum L2 norm.
        element = self.buckets[min_l2_norm_bucket][min_l2_pos_in_bucket]
        # Compute the winning probability
        incoming_l2_norm = utils.normalized_l2_norm(value, self.normalize_values)
        winning_probability = incoming_l2_norm / (incoming_l2_norm + min_l2_norm)
        rand_number = np.random.uniform(low=0, high=1)
        # update the key and value according to probability
        if rand_number < winning_probability:
            element["key"] = key
            for i in range(self.value_count):
                element["value"][i] /= winning_probability
        else:
            for i in range(self.value_count):
                element["value"][i] /= (1 - winning_probability)

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


# Tests of HyperUss
if __name__ == '__main__':
    hyperUSS = HyperUSS({"hash_function_nums": 2, "value_count": 5, "bucket_num": 200})
    groundTruth = ground_truth.GroundTruth({"value_count": 5})
    with open("./synthetic_dataset/synthetic_dataset.txt") as f:
        line = f.readline()
        while line:
            row = line.split()
            key = row[0]
            value = row[1:]
            value = [int(x) for x in value]
            hyperUSS.insert(int(key), value)
            groundTruth.insert(int(key), value)
            line = f.readline()
        f.close()
    result = hyperUSS.all_query()
    result2 = groundTruth.all_query()
    print(result[1])
    print(result2[1])
