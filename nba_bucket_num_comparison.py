from evaluation import  metrics
import pandas as pd
import numpy as np
from hyperuss import HyperUSS
from coco_imp2 import Coco
from uss import OurUSS
from ground_truth import GroundTruth
from test_nba import get_average

value_count = 3
bucket_nums = [5, 10, 15, 20, 25]
# bucket_nums = [10]
hash_num = 2

hyper_uss_metric = {'aae':[],'are':[],'f1':[]}
cocosketch_metric =  {'aae':[],'are':[],'f1':[]}
uss_metric =  {'aae':[],'are':[],'f1':[]}

df = pd.read_csv('games.csv')

#preprocess data
df.drop(df[np.isnan(df['PTS_home'])].index, inplace=True)
df.drop(df[np.isnan(df['FG_PCT_home'])].index, inplace=True)
df.drop(df[np.isnan(df['FT_PCT_home'])].index, inplace=True)

for bucket_num in bucket_nums:
    #define algorithms
    hyper_uss = HyperUSS({"hash_function_nums": hash_num, "value_count": value_count, "bucket_num": bucket_num, "normalization": False})
    cocosketch = [Coco({"hash_function_nums": hash_num, "bucket_num": bucket_num}) for _ in range(value_count)]
    uss = [OurUSS({"hash_function_nums": hash_num, "value_count": value_count,
                    "bucket_num": bucket_num, "normalization": False}) for _ in range(value_count)]
    groundTruth = GroundTruth({"value_count": value_count, "normalization": False})
    for index, row in df.iterrows():
        key = row['TEAM_ID_home']
        values = [row['PTS_home'], row['FG_PCT_home'], row['FT_PCT_home']]
        hyper_uss.insert(key, values)
        for i in range(len(values)):
            cocosketch[i].insert(key, values[i])
            uss[i].insert(key, values[i])
        groundTruth.insert(key, values)

    #query result
    gt_result, gt_a = groundTruth.all_query()
    # get_average(gt_result)
    hyperuss_result, hyper_a = hyper_uss.all_query()
    # get_average(hyperuss_result)
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
    # get_average(cocosktech_result)
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
    # get_average(uss_result)
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

datatset = 'nba'
# key = 'are'
# result = pd.DataFrame({'bucket_num': bucket_nums, 'cocosketch':cocosketch_metric[key], 'hyeper_uss':hyper_uss_metric[key],'uss':uss_metric[key]})
# result.to_csv(f'{datatset}_bucket_num_{key}.csv')
for key in cocosketch_metric.keys():
    result = pd.DataFrame({'bucket_num': bucket_nums, 'cocosketch':cocosketch_metric[key], 'hyeper_uss':hyper_uss_metric[key],'uss':uss_metric[key]})
    result.to_csv(f'{datatset}_bucket_num_{key}.csv')
   
    
            



