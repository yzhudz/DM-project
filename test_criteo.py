import pandas as pd

import ground_truth
from hyperuss import HyperUSS
from uss import OurUSS
from evaluation import metrics
from coco_imp2 import Coco

df = pd.read_csv('/Users/yifan/Downloads/dac/test.txt', sep='\t', header=None)

numeric_columns = df.iloc[:, :13]
categorical_columns = df.iloc[:, 13:]

numeric_columns.fillna(numeric_columns.mean(), inplace=True)
categorical_columns.fillna(method='ffill', axis=1, inplace=True)
numeric_columns = numeric_columns.applymap(lambda x: max(x, 0))

df = pd.concat([numeric_columns, categorical_columns], axis=1)

hyperUSS = HyperUSS({"hash_function_nums": 2, "value_count": 13, "bucket_num": 500, "normalization": True})
cocoSketch = [Coco({"hash_function_nums": 2, "bucket_num": 500}) for _ in range(13)]
uss = [OurUSS({"hash_function_nums": 2, "value_count": 13,
                  "bucket_num": 500, "normalization": False}) for _ in range(13)]
groundTruth = ground_truth.GroundTruth({"value_count": 13, "normalization": True})

counter = 0
for _, row in df.iterrows():
    key = int(row[13], 16)
    value = row[:13].tolist()
    hyperUSS.insert(key, value)
    for i, coco in enumerate(cocoSketch):
        coco.insert(key, value[i])
        uss.insert(key, value[i])
    groundTruth.insert(key, value)
    counter += 1
    if counter % 1000 == 0:
        print(counter)


result, a = hyperUSS.all_query()
result2, a2 = groundTruth.all_query()

# query
result3 = {}
for i in range(13):
    single_result = cocoSketch[i].all_query()
    for k, v in single_result.items():
        if k not in result3:
            result3[k] = [0] * 13
        result3[k][i] = single_result[k]

result4 = {}
for i in range(13):
    single_result = uss[i].all_query()
    for k, v in single_result.items():
        if k not in result3:
            result4[k] = [0] * 13
        result4[k][i] = single_result[k]


print("Hyper USS: ")
print('f1 aae, are = ', metrics(result, result2, a2))


print("CocoSketch: ")
print('f1 aae, are = ', metrics(result1, result2, a2))

print("USS: ")
print('f1 aae, are = ', metrics(result4, result2, a2))