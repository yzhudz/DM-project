# Cocosketch
from typing import Dict, List
from sketch import Sketch
import mmh3
import utils as utils
import numpy as np
import ground_truth
import random

# Single.h
class Single(object):
    def __init__(self, id, value):
        self.id = id
        self.value = value

class OurSingle:
    def __init__(self, hash_num=2, bucket_num=100, name="OurSingle"):
        self.name = name
        self.hash_num = hash_num
        self.bucket_num = bucket_num
        self.counter =  [[Single(0, 0) for _ in range(self.bucket_num)] for _ in range(hash_num)]

    def insert(self, item):
        minimum = float('inf')
        minPos, minHash = None, None

        for i in range(self.hash_num):
            position = mmh3.hash(item.id.to_bytes(length=8, byteorder='big'), seed=i) % self.bucket_num
            if self.counter[i][position].id == item.id:
                self.counter[i][position].value += item.value
                return 
            if abs(self.counter[i][position].value) < minimum:
                minPos, minHash = position, i
                minimum = abs(self.counter[i][position].value)

        if item.value == 0:
            return 

        self.counter[minHash][minPos].value +=item.value
        rand_number = np.random.uniform(low=0, high=1)
        if rand_number < abs(item.value) / abs(self.counter[minHash][minPos].value):
               self.counter[minHash][minPos].id = item.id
        
    def all_query(self):
        ret = {}
        for i in range(self.hash_num):
            for j in range(self.bucket_num):
                ret[self.counter[i][j].id] = self.counter[i][j].value
        return ret
    
if __name__ == '__main__':
    value_count = 5
    cocosketch = [OurSingle() for _ in range(value_count)]
    groundTruth = ground_truth.GroundTruth({"value_count": 5})
    with open("synthetic_dataset.txt") as f:
        line = f.readline()
        while line:
            row = line.split()
            key = row[0]
            value = row[1:]
            value = [int(x) for x in value]
            for i in range(value_count):
                cocosketch[i].insert(Single(int(key), value[i]))
                
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
    print(results[10])
    print(result2[10])