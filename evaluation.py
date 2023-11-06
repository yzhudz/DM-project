from typing import Dict, List

import numpy as np
import pandas as pd

import utils


# estimation = pd.read_csv('estimation.csv')
# ground_truth = pd.read_csv('ground_truth.csv')


def f1_score(estimation: Dict[int, List[float]], ground_truth: Dict[int, List[float]], total: List[float], alpha=0.00001) -> float:
    tp, tn, fp, fn = 0, 0, 0, 0
    for key, values in ground_truth.items():
        for index, value in enumerate(values):
            v = value
            if key in estimation:
                est_v = estimation[key][index]
            else:
                est_v = -1
            real_frequent = v > alpha * total[index]
            est_frequent = est_v > alpha * total[index]
            if real_frequent and est_frequent:
                tp += 1
            elif not real_frequent and est_frequent:
                fp += 1
            elif real_frequent and not est_frequent:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# AAE and ARE
def aae_and_are(estimation, ground_truth):
    estimation.columns = [str(x) for x in estimation.columns.to_list()]
    ground_truth.columns = [str(x) for x in ground_truth.columns.to_list()]
    list_key_estimated=estimation['index'].to_list()
    fS_in_Phi = ground_truth[ground_truth['index'].apply(lambda x: x in list_key_estimated)].sort_values(by='index').reset_index(drop=True)
    Phi = estimation.sort_values(by='index').reset_index(drop=True)
    # aae = np.sum(np.sum(np.abs(Phi - fS_in_Phi), axis=1), axis=0) / Phi.shape[0]
    aae = np.mean(np.mean(np.abs(Phi.iloc[:, 1:] - fS_in_Phi.iloc[:, 1:])))
    # are = aae / np.sum(np.sum(fS_in_Phi.iloc[:, 1:], axis=1), axis=0)
    are = np.nanmean(
        np.abs(Phi.iloc[:, 1:] - fS_in_Phi.iloc[:, 1:]) / fS_in_Phi.iloc[:, 1:].mask(fS_in_Phi.iloc[:, 1:].eq(0)))
    return aae, are


# def f1_score_coco_uss(estimation, ground_truth):
#     estimation.columns = [str(x) for x in estimation.columns.to_list()]
#     ground_truth.columns = [str(x) for x in ground_truth.columns.to_list()]
#     list_key_estimated = estimation.sort_values(by='0', ascending=False).iloc[:, 0].to_list()
#     list_key_ground_truth_top_n = ground_truth.sort_values(by='0', ascending=False).iloc[0:estimation.shape[0],
#                                   0].to_list()
#     list_key_ground_truth = ground_truth['index'].to_list()
#     out_estimation = []
#     for i in range(len(list_key_ground_truth)):
#         if list_key_ground_truth[i] not in list_key_estimated:
#             out_estimation.append(list_key_ground_truth[i])
#     tp = 0
#     for i in range(len(list_key_estimated)):
#         if list_key_estimated[i] in list_key_ground_truth_top_n:
#             tp += 1
#     fp = len(list_key_estimated) - tp
#
#     fn = 0
#     for i in range(len(out_estimation)):
#         if out_estimation[i] in list_key_ground_truth_top_n:
#             fn += 1
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     f1 = 2 * precision * recall / (precision + recall)
#     return f1
#
