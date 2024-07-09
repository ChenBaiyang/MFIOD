import os, time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import minmax_scale
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

    for idx, dataset in enumerate(dataset_list):
        inputs = np.load(data_dir + dataset)
        data = inputs['X']
        n, m = data.shape
        label = inputs['y']

        # provide nominal attributs otherwise all attributes are treated as numerical values
        try:
            nominals = inputs['nominals']
        except:
            nominals = np.full(m, False)
        print("Dataset:{}\tShape:{}\t#Outlier:{}\t#Nominals:{}".format(dataset[:-4], (n, m), label.sum(), nominals.sum()))

        # Min-max scale on numerical attributes
        numericals = np.logical_not(nominals)
        if numericals.sum() > 0:
            data[:, numericals] = minmax_scale(data[:, numericals])

        # run with default parameter
        t0 = time.time()
        model = MFIOD(data, nominals, lambs='Default')
        out_scores = model.detection()
        paras = model.lambs
        t1 = time.time()

        Pt, auc, pr = evaluation_Pt_auc_pr(out_scores, label)
        print(dataset[:-4], 'MFIOD', str(paras), 'Pt:', round(Pt*100,3), 'AUC:', round(auc*100,3), 'Pr:', round(pr*100,3), "Time:", round(t1-t0,2))

        # save to csv file
        scores = [dataset[:-4], 'MFIOD', str(paras), str(Pt)[:8], str(auc)[:8], str(pr)[:8]]
        open('result_MFIOD_new_datasets.csv', 'a').write(','.join(scores) + '\n')


