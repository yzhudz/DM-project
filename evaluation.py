import numpy as np
import pandas as pd

# estimation = pd.read_csv('estimation.csv')
# ground_truth = pd.read_csv('ground_truth.csv')


def f1_score(estimation, ground_truth):
    estimation.columns=['Unnamed: 0','0','1','2','3','4']
    ground_truth.columns = ['Unnamed: 0', '0', '1', '2', '3', '4']
    list_key_estimated = estimation.sort_values(by='0', ascending=False).iloc[0:200, 0].to_list()
    list_key_ground_truth_top_n = ground_truth.sort_values(by='0', ascending=False).iloc[0:200, 0].to_list()
    list_key_ground_truth = ground_truth['Unnamed: 0'].to_list()
    out_estimation = []
    for i in range(len(list_key_ground_truth)):
        if list_key_ground_truth[i] not in list_key_estimated:
            out_estimation.append(list_key_ground_truth[i])
    tp = 0
    for i in range(len(list_key_estimated)):
        if list_key_estimated[i] in list_key_ground_truth_top_n:
            tp += 1
    fp = len(list_key_estimated) - tp

    fn = 0
    for i in range(len(out_estimation)):
        if out_estimation[i] in list_key_ground_truth_top_n:
            fn += 1
    tn = len(out_estimation) - fn
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# AAE and ARE
def aae_and_are(estimation,ground_truth):
    estimation.columns = ['Unnamed: 0', '0', '1', '2', '3', '4']
    ground_truth.columns = ['Unnamed: 0', '0', '1', '2', '3', '4']
    list_key_estimated = estimation.sort_values(by='0', ascending=False).iloc[0:200, 0].to_list()
    fS_in_Phi = ground_truth[ground_truth['Unnamed: 0'].apply(lambda x: x in list_key_estimated)].sort_values(
        by='Unnamed: 0').reset_index(drop=True)
    Phi = estimation.sort_values(by='Unnamed: 0').reset_index(drop=True)
    aae = np.sum(np.sum(np.abs(Phi - fS_in_Phi),axis=1),axis=0) / Phi.shape[0]
    are = aae / np.sum(np.sum(fS_in_Phi.iloc[:, 1:],axis=1),axis=0)
    return aae, are
