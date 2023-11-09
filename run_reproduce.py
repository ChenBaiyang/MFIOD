import os
import time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from models import MFIOD


def evaluation_Pt_auc_pr(out_scores, label):
    order = np.argsort(out_scores)
    for t in [label.sum()]:
        pos_idx = order[-t:]
        result = np.zeros_like(label)
        result[pos_idx] = True
        P, R, F1, _ = precision_recall_fscore_support(label, result, average='binary')
    auc = roc_auc_score(label, out_scores)
    pr = average_precision_score(y_true=label, y_score=out_scores, pos_label=1)
    return P, auc, pr

if __name__ == "__main__":
    data_dir = 'data/'
    dataset_list = os.listdir(data_dir)
    lambdas = [[0.95],
               [0.2, 0.75],
               [.9],
               [0.05, 0.9, 0.95],
               [.1],
               [0.8, 0.9, 0.9],
               [.5],
               [.55],
               [.4],
               [0.65, 0.65, 0.95],
               [0.85, 0.9, 0.9],
               [.55],
               [.4],
               [.75],
               [0.3, 0.7]]

    for idx, dataset in enumerate(dataset_list):
        d = np.load(data_dir + dataset)
        data = d['X']
        n, m = data.shape
        label = d['y']
        nominals = d['nominals']
        print("Dataset:{}\tShape:{}\t#Outlier:{}\t#Nominals:{}".format(dataset[:-4], (n, m), label.sum(), nominals.sum()))

        t0 = time.time()
        lambs = lambdas[idx]
        out_scores = MFIOD(data, nominals, lambs=lambs, n_bins=20).detection()
        Pt, auc, pr = evaluation_Pt_auc_pr(out_scores, label)
        t1 = time.time()
        print('Pt:', round(Pt*100,3), 'AUC:', round(auc*100,3), 'Pr:', round(pr*100,3), "Time:", round(t1-t0,2))
        # scores = [dataset[:-4], 'MFIOD', 'np', str(Pt)[:8], str(auc)[:8], str(pr)[:8]]
        # open('result_MFIOD_reproduce.csv', 'a').write(','.join(scores) + '\n')

