from evaluation import metrics
import pandas as pd
import numpy as np
import csv
from hyperuss import HyperUSS
from coco_imp2 import Coco
from uss import OurUSS
from ground_truth import GroundTruth

def get_average(result, avg_index=[1,2]):
    for k in result.keys():
        for i in avg_index:
            if result[k][-1] != 0:
                result[k][i] /= result[k][-1]
            else:
                result[k][i] = 0
    
            

value_count = 4
bucket_num = 10
hash_num = 2

#read file
df = pd.read_csv('games.csv')

#preprocess data
df.drop(df[np.isnan(df['PTS_home'])].index, inplace=True)
df.drop(df[np.isnan(df['FG_PCT_home'])].index, inplace=True)
df.drop(df[np.isnan(df['FT_PCT_home'])].index, inplace=True)

#define algorithms
hyper_uss = HyperUSS({"hash_function_nums": hash_num, "value_count": value_count, "bucket_num": bucket_num, "normalization": False})
cocosketch = [Coco({"hash_function_nums": hash_num, "bucket_num": bucket_num}) for _ in range(value_count)]
uss = [OurUSS({"hash_function_nums": hash_num, "value_count": value_count,
                  "bucket_num": bucket_num, "normalization": False}) for _ in range(value_count)]
groundTruth = GroundTruth({"value_count": value_count, "normalization": False})
for index, row in df.iterrows():
    key = row['TEAM_ID_home']
    values = [row['PTS_home'], row['FG_PCT_home'], row['FT_PCT_home'], 1]
    hyper_uss.insert(key, values)
    for i in range(len(values)):
        cocosketch[i].insert(key, values[i])
        uss[i].insert(key, values[i])
    groundTruth.insert(key, values)

#query result
gt_result, gt_a = groundTruth.all_query()
get_average(gt_result)
print("*"*30)
print("testing result of hyper_uss")
hyperuss_result, hyper_a = hyper_uss.all_query()
get_average(hyperuss_result)
print('f1 aae, are = ', metrics(hyperuss_result, gt_result, gt_a))
print("*"*30)
print("testing result of cocosketch")
cocosktech_result = {}
for i in range(value_count):
    single_result = cocosketch[i].all_query()
    for k, v in single_result.items():
        if k not in cocosktech_result:
            cocosktech_result[k] = [0] * value_count
        cocosktech_result[k][i] = single_result[k]
get_average(cocosktech_result)
print('f1 aae, are = ', metrics(cocosktech_result, gt_result, gt_a))
print("*"*30)
print("testing result of uss")
uss_result = {}
for i in range(value_count):
    single_result = uss[i].all_query()
    for j, k in single_result.items():
        if j not in uss_result:
            uss_result[j] = [0] * value_count
        uss_result[j][i] = single_result[j]
get_average(uss_result)
print('f1 aae, are = ', metrics(uss_result, gt_result, gt_a))