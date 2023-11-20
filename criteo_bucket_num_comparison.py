import pandas as pd

import ground_truth
from hyperuss import HyperUSS
from uss import OurUSS
from evaluation import metrics
from coco_imp2 import Coco

df = pd.read_csv('./test.txt', sep='\t', header=None)

numeric_columns = df.iloc[:, :13]
categorical_columns = df.iloc[:, 13:]

numeric_columns.fillna(numeric_columns.mean(), inplace=True)
categorical_columns.fillna(method='ffill', axis=1, inplace=True)
numeric_columns = numeric_columns.applymap(lambda x: max(x, 0))

df = pd.concat([numeric_columns, categorical_columns], axis=1)


hyper_uss_metric = {'aae':[],'are':[],'f1':[]}
cocosketch_metric =  {'aae':[],'are':[],'f1':[]}
uss_metric =  {'aae':[],'are':[],'f1':[]}

bucket_nums = [300, 500, 1000, 2000]
value_count = 13
hash_num = 2


for bucket_num in bucket_nums:
    #define algorithms
    hyper_uss = HyperUSS({"hash_function_nums": hash_num, "value_count": value_count, "bucket_num": bucket_num, "normalization": False})
    cocosketch = [Coco({"hash_function_nums": hash_num, "bucket_num": bucket_num}) for _ in range(value_count)]
    uss = [OurUSS({"hash_function_nums": hash_num, "value_count": value_count,
                    "bucket_num": bucket_num, "normalization": False}) for _ in range(value_count)]
    groundTruth = ground_truth.GroundTruth({"value_count": value_count, "normalization": True})

    counter = 0
    for _, row in df.iterrows():    
        key = int(row[13], 16)
        value = row[:13].tolist()
        hyper_uss.insert(key, value)
        for i in range(value_count):
            coco = cocosketch[i]
            u = uss[i]
            coco.insert(key, value[i])
            u.insert(key, value[i])
        groundTruth.insert(key, value)
        counter += 1
        if counter % 100000 == 0:
            print(counter)
 

    #query result
    gt_result, gt_a = groundTruth.all_query()
    hyperuss_result, hyper_a = hyper_uss.all_query()
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

datatset = 'criteo'
for key in cocosketch_metric.keys():
    result = pd.DataFrame({'bucket_num': bucket_nums, 'cocosketch':cocosketch_metric[key], 'hyeper_uss':hyper_uss_metric[key],'uss':uss_metric[key]})
    result.to_csv(f'{datatset}_bucket_num_{key}.csv')
   
    
            



