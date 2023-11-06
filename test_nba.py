from evaluation import aae_and_are, f1_score_coco_uss
import pandas as pd
import numpy as np
import csv
from hyperuss import HyperUSS
from coco_imp2 import Coco
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
groundTruth = GroundTruth({"value_count": value_count})
for index, row in df.iterrows():
    key = row['TEAM_ID_home']
    values = [row['PTS_home'], row['FG_PCT_home'], row['FT_PCT_home'], 1]
    hyper_uss.insert(key, values)
    for i in range(len(values)):
        cocosketch[i].insert(key, values[i])
    groundTruth.insert(key, values)

#query result
gt_result = groundTruth.all_query()
get_average(gt_result)
print("*"*30)
print("testing result of hyper_uss")
hyperuss_result = hyper_uss.all_query()
get_average(hyperuss_result)
print('f1 score: ', f1_score_coco_uss(pd.DataFrame(gt_result).T.reset_index(), pd.DataFrame(hyperuss_result).T.reset_index()))
aae, are = aae_and_are(pd.DataFrame(gt_result).T.reset_index(), pd.DataFrame(hyperuss_result).T.reset_index())
print('AAE: ', aae)
print('ARE: ', are)
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
print('f1 score: ', f1_score_coco_uss(pd.DataFrame(gt_result).T.reset_index(), pd.DataFrame(cocosktech_result).T.reset_index()))
aae, are = aae_and_are(pd.DataFrame(gt_result).T.reset_index(), pd.DataFrame(cocosktech_result).T.reset_index())
print('AAE: ', aae)
print('ARE: ', are)