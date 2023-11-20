from evaluation import  metrics
import pandas as pd
import numpy as np
from hyperuss import HyperUSS
from coco_imp2 import Coco
from uss import OurUSS
from ground_truth import GroundTruth

value_count = 5
bucket_nums = [500, 1000, 1500, 2000]
hash_num = 2

hyper_uss_metric = {'aae':[],'are':[],'f1':[]}
cocosketch_metric =  {'aae':[],'are':[],'f1':[]}
uss_metric =  {'aae':[],'are':[],'f1':[]}

for bucket_num in bucket_nums:
    #define algorithms
    hyper_uss = HyperUSS({"hash_function_nums": hash_num, "value_count": value_count, "bucket_num": bucket_num, "normalization": True})
    cocosketch = [Coco({"hash_function_nums": hash_num, "bucket_num": bucket_num}) for _ in range(value_count)]
    uss = [OurUSS({"hash_function_nums": hash_num, "value_count": value_count,
                    "bucket_num": bucket_num, "normalization": False}) for _ in range(value_count)]
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
                uss[i].insert(int(key), value[i])
            hyper_uss.insert(int(key), value)
            groundTruth.insert(int(key), value)
            line = f.readline()
        f.close()

    #query result
    gt_result, gt_a = groundTruth.all_query()
    hyperuss_result, hyper_a = hyper_uss.all_query()
    # print('f1 aae, are = ', metrics(hyperuss_result, gt_result, gt_a))
    f1, aae, are = metrics(hyperuss_result, gt_result, gt_a)
    hyper_uss_metric['f1'].append(f1)
    hyper_uss_metric['aae'].append(aae)
    hyper_uss_metric['are'].append(are)

    cocosktech_result = {}
    for i in range(value_count):
        single_result = cocosketch[i].all_query()
        for k, v in single_result.items():
            if k not in cocosktech_result:
                cocosktech_result[k] = [0] * value_count
            cocosktech_result[k][i] = single_result[k]
    # print('f1 aae, are = ', metrics(cocosktech_result, gt_result, gt_a))
    f1, aae, are = metrics(cocosktech_result, gt_result, gt_a)
    cocosketch_metric['f1'].append(f1)
    cocosketch_metric['aae'].append(aae)
    cocosketch_metric['are'].append(are)
    
    uss_result = {}
    for i in range(value_count):
        single_result = uss[i].all_query()
        for j, k in single_result.items():
            if j not in uss_result:
                uss_result[j] = [0] * value_count
            uss_result[j][i] = single_result[j]
    # print('f1 aae, are = ', metrics(uss_result, gt_result, gt_a))
    f1, aae, are = metrics(uss_result, gt_result, gt_a)
    uss_metric['f1'].append(f1)
    uss_metric['aae'].append(aae)
    uss_metric['are'].append(are)

print("*"*30)
print("testing result of hyper_uss")
print(hyper_uss_metric)

print("*"*30)
print("testing result of cocosketch")
print(cocosketch_metric)


print("*"*30)
print("testing result of uss")
print(uss_metric)

# #format
# aee_result = pd.DataFrame('bucket_num':bucket_nums, 'cocosketch'cocosketch_metric['aae']:, 'hyeper_uss':hyper_uss_metric['aae'],'uss':uss_metric['aae'])
# aee_result.to_csv('buckert_num_aee_result.csv')

# are_result = pd.DataFrame('bucket_num':bucket_nums, 'cocosketch'cocosketch_metric['are']:, 'hyeper_uss':hyper_uss_metric['are'],'uss':uss_metric['are'])

# aee_result = pd.DataFrame('bucket_num':bucket_nums, 'cocosketch'cocosketch_metric['aae']:, 'hyeper_uss':hyper_uss_metric['aae'],'uss':uss_metric['aae']) 

for key in cocosketch_metric.keys():
    result = pd.DataFrame({'bucket_num': bucket_nums, 'cocosketch':cocosketch_metric[key], 'hyeper_uss':hyper_uss_metric[key],'uss':uss_metric[key]})
    result.to_csv(f'buckert_num_{key}_result.csv')