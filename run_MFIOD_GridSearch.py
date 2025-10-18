import os, time
import numpy as np
from sklearn.preprocessing import minmax_scale
from models import MFIOD, evaluation_Pt_auc_pr


if __name__ == "__main__":
    data_dir = 'data/'
    dataset_list = os.listdir(data_dir)
    result_file_name = 'result_MFIOD_gridSearch.csv'
    open(result_file_name, 'w').write('dataset,model,para,Pt,auc,pr,time\n')

    for idx, dataset in enumerate(dataset_list):
        d = np.load(data_dir + dataset)
        data = d['X']
        label = d['y']
        n, m = data.shape

        try:
            nominals = d['nominals']
        except:
            print("\tNominal attributes not designated, all attributes treated as numeric.")
            nominals = np.full(m, False, dtype=np.bool_)

        print("Dataset:\t{}\t\tShape:{}\t#Outlier:{}\t#Nominals:{}".format(dataset[:-4], (n, m), label.sum(), nominals.sum()))

        # Min-max scale on numerical attributes
        numericals = np.logical_not(nominals)
        if numericals.sum() > 0:
            data[:, numericals] = minmax_scale(data[:, numericals])

        # run with grid-searching parameters
        t0 = time.time()
        paras = {'lambs': [0.1, 0.3, 0.5, 0.7, 0.9]}
        # Only use 1000 train samples to perform grid search
        train_X = data[:1000]
        train_y = label[:1000]
        model = MFIOD(data, nominals, lambs='Grid_search', train_X=train_X, train_y=train_y, paras=paras['lambs'])
        out_scores = model.detection()
        t1 = time.time()


        Pt, auc, pr = evaluation_Pt_auc_pr(out_scores, label)
        print("\tLambdas:{}\t".format(model.lambs), 'Pt:', round(Pt*100,3), 'AUC:', round(auc*100,3), 'Pr:', round(pr*100,3), "Time:", round(t1-t0,1))

        # save to csv file
        scores = [dataset[:-4], 'MFIOD', str(model.lambs).replace(',', ' '), str(Pt)[:8], str(auc)[:8], str(pr)[:8], str(t1-t0)]
        open(result_file_name, 'a').write(','.join(scores) + '\n')


