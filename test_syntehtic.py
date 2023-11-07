from evaluation import  metrics
import pandas as pd
import numpy as np
from hyperuss import HyperUSS
from coco_imp2 import Coco
from ground_truth import GroundTruth

value_count = 5
bucket_num = 1000
hash_num = 2


#define algorithms
hyper_uss = HyperUSS({"hash_function_nums": hash_num, "value_count": value_count, "bucket_num": bucket_num, "normalization": True})
cocosketch = [Coco({"hash_function_nums": hash_num, "bucket_num": bucket_num}) for _ in range(value_count)]
groundTruth = GroundTruth({"value_count": value_count,"normalization": True})
with open("./synthetic_dataset/synthetic_dataset.txt") as f:
    line = f.readline()
    while line:
        row = line.split()
        key = row[0]
        value = row[1:]
        value = [int(x) for x in value]
        for i in range(value_count):
            cocosketch[i].insert(int(key), value[i])
        hyper_uss.insert(int(key), value)
        groundTruth.insert(int(key), value)
        line = f.readline()
    f.close()

#query result
gt_result, gt_a = groundTruth.all_query()
print("*"*30)
print("testing result of hyper_uss")
hyperuss_result, hyper_a = hyper_uss.all_query()
print('f1 score: ', metrics(hyperuss_result, gt_result, gt_a))
print("*"*30)
print("testing result of cocosketch")
cocosktech_result = {}
for i in range(value_count):
    single_result = cocosketch[i].all_query()
    for k, v in single_result.items():
        if k not in cocosktech_result:
            cocosktech_result[k] = [0] * value_count
        cocosktech_result[k][i] = single_result[k]
print('f1 aae, are = ', metrics(cocosktech_result, gt_result, gt_a))
