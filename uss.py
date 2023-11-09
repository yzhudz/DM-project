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


class Node():
    def __init__(self):
        self.id = 0
        self.value_list = []


class OurUSS():

    def __init__(self):
        # self.file_path = file_path
        # super().__init__(params)
        # self.stream_item_set = []
        self.bucket_size = 100
        self.current_size = 0
        self.bucket = {}
        self.error = 0
        # self.exact_counter = defaultdict(int)
        # self.space_saving_count = defaultdict(int)
        # self.amount_of_words = 0

    def insert(self, key, value):
        if key in self.bucket:
            arr1 = np.array(self.bucket[key][2])
            arr2 = np.array(value)
            self.bucket[key][0] += 1
            self.bucket[key][2] = list(arr1 + arr2)
        elif len(self.bucket) + 1 <= self.bucket_size:
            self.bucket[key] = [1, self.error, value]
        else:
            min_error = float(np.inf)
            for i in self.bucket:
                # print(i)
                if self.bucket[i][0] + self.bucket[i][1] < min_error:
                    min_error = self.bucket[i][0] + self.bucket[i][1]
            self.error = min_error
            keys = [v for i, v in enumerate(self.bucket) if self.bucket[v][0] + self.bucket[v][1] <= self.error]
            for key in keys:
                if random.uniform(0, 1) < 1 / (self.error + 1):
                    self.bucket.pop(key)
            self.bucket[key] = [1, self.error, value]

    def all_query(self):
        self.bucket = dict(
            sorted(self.bucket.items(), key=lambda i: i[1][0] + i[1][1], reverse=True))
        result = {i: self.bucket[i][2] for i in self.bucket}
        return result

    def read_file(self):
        with open('synthetic_dataset/synthetic_dataset.txt') as f:
            for line in f:
                row = line.split()
                value = [int(x) for x in row]
                self.stream_item_set.append(value)
        # print(self.words)

    # def space_saving_counter(self, k):
    #     """ Space saving counter """
    #     counters = {}
    #     for word in self.words:
    #         if word in counters:
    #             counters[word] += 1
    #         elif len(counters) + 1 < k:
    #             counters[word] = 1
    #         else:
    #             min_counter = min(counters, key=counters.get)
    #             counters[word] = counters.pop(min_counter) + 1
    #     self.space_saving_count = counters
    #     self.space_saving_count = dict(sorted(self.space_saving_count.items(), key=lambda item: item[1], reverse=True))

    def unbiased_space_saving_counter(self, k):
        """ Space saving counter """
        counters = {}
        for word in self.stream_item_set:
            key = word[0]
            if key in counters:
                counters[key] += 1
            elif len(counters) + 1 <= k:
                counters[key] = 1
            else:
                min_counter_key = min(counters, key=counters.get)
                if random.uniform(0, 1) < 1 / (min_counter_key + 1):
                    counters[key] = counters.pop(min_counter_key) + 1
                # else:
                #     counters[min_counter_key] += 1
            # print(values)
        self.space_saving_count = counters
        self.space_saving_count = dict(sorted(self.space_saving_count.items(), key=lambda item: item[1], reverse=True))
        print(self.space_saving_count)
        print(len(self.space_saving_count))

    def unbiased_space_saving_value(self, k):
        """ Space saving counter """
        bucket = {}
        error = 0
        for item in self.stream_item_set:
            key = item[0]
            value = item[1:]
            if key in bucket:
                arr1 = np.array(bucket[key][2])
                arr2 = np.array(value)
                bucket[key][0] += 1
                bucket[key][2] = list(arr1 + arr2)
            elif len(bucket) + 1 <= k:
                bucket[key] = [1, error, value]
            else:
                min_error = float(np.inf)
                for i in bucket:
                    # print(i)
                    if bucket[i][0] + bucket[i][1] < min_error:
                        min_error = bucket[i][0] + bucket[i][1]
                error = min_error
                keys = [v for i, v in enumerate(bucket) if bucket[v][0] + bucket[v][1] <= error]
                for key in keys:
                    if random.uniform(0, 1) < 1 / (error + 1):
                        bucket.pop(key)
                bucket[key] = [1, error, value]

        bucket = dict(
            sorted(bucket.items(), key=lambda i: i[1][0] + i[1][1], reverse=True))
        print(bucket)
        # print(self.space_saving_count)
        # print(values)


if __name__ == "__main__":
    uss = OurUSS()
    # uss.read_file()
    groundTruth = ground_truth.GroundTruth({"value_count": 5, "normalization": False})
    with open('synthetic_dataset/synthetic_dataset.txt') as f:
        for line in f:
            row = line.split()
            value = [int(x) for x in row]
            # uss.stream_item_set.append(value)
            uss.insert(value[0], value[1:])
            groundTruth.insert(value[0], value[1:])

    result = uss.all_query()
    result2, truth_a_value = groundTruth.all_query()
    print('f1 score: ', metrics(result, result2, truth_a_value))

    # uss.unbiased_space_saving_counter(100)
    # uss.unbiased_space_saving_value(100)
