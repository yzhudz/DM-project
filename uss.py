import sys
from typing import Dict, List
from sketch import Sketch
import random
import mmh3
import utils as utils
import numpy as np
import ground_truth


class OurUSS():

    def __init__(self):
        # self.file_path = file_path
        self.words = []
        # self.exact_counter = defaultdict(int)
        # self.space_saving_count = defaultdict(int)
        # self.amount_of_words = 0

    def insert(self, key, value):
        # get the arguments
        self.params["key_count"]
        pass

    def all_query(self):
        pass

    def read_file(self):
        with open('synthetic_dataset/synthetic_dataset.txt') as f:
            for line in f:
                row = line.split()
                value = [int(x) for x in row]
                self.words.append(value)
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
        for word in self.words:
            if word[0] in counters:
                counters[word[0]] += 1
            elif len(counters) + 1 < k:
                counters[word[0]] = 1
            else:
                min_counter = min(counters, key=counters.get)
                if random.uniform(0, 1) < 1 / (min_counter + 1):
                    counters[word[0]] = counters.pop(min_counter) + 1
                else:
                    counters[min_counter] += 1
        self.space_saving_count = counters
        self.space_saving_count = dict(sorted(self.space_saving_count.items(), key=lambda item: item[1], reverse=True))
        print(self.space_saving_count)


if __name__ == "__main__":
    uss = OurUSS()
    uss.read_file()
    uss.unbiased_space_saving_counter(100)
    # uss.unbiased_space_saving_counter(100)
