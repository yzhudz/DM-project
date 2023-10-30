import sys
from typing import Dict, List
from sketch import Sketch
import mmh3
import utils as utils
import numpy as np
import ground_truth
import random


class Coco(Sketch):

    def __init__(self, params: Dict[str, int]):
        super().__init__(params)
        # d hash functions (bucket arrays)
        self.hash_function_num = params["hash_function_nums"]
        # w buckets in each array
        self.bucket_num = params["bucket_num"]
        # d * w buckets
        self.buckets = [[{"key": -1, "value": 0} for _ in range(self.bucket_num)] for _ in
                        range(self.hash_function_num)]

    def insert(self, key: int, value: int):
        """
        Insert an element into HyperUSS sketch.

        :param key: the identifier(key) of the element
        :param value: attribute values of the element.
        """  
        min_value = float('inf')
        min_bucket, min_pos_in_bucket = -1, -1

        for i in range(self.hash_function_num):
            bucket_i_pos = mmh3.hash(key.to_bytes(length=8, byteorder='big'), seed=i) % self.bucket_num
            element = self.buckets[i][bucket_i_pos]
            # Element key is recorded in one of the buckets, simply increase the value.
            if element["key"] == key:
                element["value"] += value
                return
            # An empty slot is found, first record the position.
            if element["key"] == -1:
                element = self.buckets[i][bucket_i_pos]
                element["key"] = key
                element["value"] = value
                return
            val = abs(element['value'])
            if val < min_value:
                min_bucket, min_pos_in_bucket = i, bucket_i_pos
                min_value = val
        
        if value == 0:
            return
        
        # add the value to the minimum bucket
        self.buckets[min_bucket][min_pos_in_bucket]['value'] += value
        # Compute the winning probability to replace the key value
        rand_number = np.random.uniform(low=0, high=1)
        if rand_number < abs(value) / abs(self.buckets[min_bucket][min_pos_in_bucket]['value']):
            self.buckets[min_bucket][min_pos_in_bucket]['key'] = key

    def all_query(self) -> Dict[int, List[int]]:
        """
        Acquire sum of all elements grouped by key.

        """
        result = {}
        for bucket in self.buckets:
            for element in bucket:
                if element["key"] == -1:
                    continue
                result[element["key"]] = element["value"]
        return result


# Tests of CocoSketch
if __name__ == '__main__':
    value_count = 5
    cocosketch = [Coco({"hash_function_nums": 2, "bucket_num": 100}) for _ in range(value_count)]
    groundTruth = ground_truth.GroundTruth({"value_count": 5})
    with open("synthetic_dataset\synthetic_dataset.txt") as f:
        line = f.readline()
        while line:
            row = line.split()
            key = row[0]
            value = row[1:]
            value = [int(x) for x in value]
            for i in range(value_count):
                cocosketch[i].insert(int(key),value[i])
                
            groundTruth.insert(int(key), value)
            line = f.readline()
        f.close()
    #query
    results = {}
    for i in range(value_count):
        single_result = cocosketch[i].all_query()
        for k, v in single_result.items():
            if k not in results:
                results[k] = []
            results[k].append(single_result[k])
    result2 = groundTruth.all_query()
    print(results[93])
    print(result2[93])

# TODO: matrix
