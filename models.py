import numpy as np
import torch as t
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


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


class MFIOD(object):
    def __init__(self, data, nominals, lambs=None, n_bins=20, train_X=None, train_y=None, paras=None):
        self.data = t.from_numpy(data).float().to(device)
        self.nominals = nominals
        self.__find_bins__(n_bins=n_bins)

        if lambs == 'Default' or lambs is None:
            self.__make_relation_matrix_for_bins__()
            # Choose the three smallest ave_distance as default parameters
            lambs_new = 1 - self.ave_dist_bins
            lambs_new.sort()
            lambs_new = lambs_new[:3].tolist()
            self.lambs = [round(i,3) for i in lambs_new]
            # print(self.lambs)
        elif lambs == 'Grid_search':
            assert train_y is not None and paras is not None, 'Grid search requires data labels'
            self.lambs = self.grid_search(train_X, train_y, paras)
            self.__make_relation_matrix_for_bins__()
        else:
            # Assume parameters are given in a list, e.g., [0.1, 0.2, 0.3]
            # print("Using given parameters:", self.lambs)
            assert 0 < min(lambs) and 1 > max(lambs), 'Parameters not allowed...'
            self.lambs = lambs
            self.__make_relation_matrix_for_bins__v0()

        self.__multi_scale_granule__(self.lambs)

    def __find_bins__(self, n_bins=20):
        data_std = self.data.cpu().std(0)
        hist, bin_edges = np.histogram(data_std, n_bins)
        bin_edges[-1] += 1e-6
        bin_indices = np.digitize(data_std, bin_edges) - 1
        self.bins = []
        for i in range(n_bins):
            indices_for_bin_i = np.where(bin_indices == i)[0]
            if len(indices_for_bin_i) > 0:
                self.bins.append(indices_for_bin_i.tolist())
        # print('Number of bins:', len(self.bins))


    def __make_relation_matrix_for_bins__v0(self, X=None):
        if X is None:
            X = self.data
        n, m = X.shape
        X = X.T.unsqueeze(-1)
        dist_matrix = t.cdist(X, X, p=1).float() # Shape (m, n, n)

        if self.nominals.sum() > 0:
            dist_matrix[self.nominals] = (dist_matrix[self.nominals] > 1e-6).float()
        self.dist_matrix = t.zeros((len(self.bins),n,n), dtype=t.float32).to(device)
        for idx, bin in enumerate(self.bins):
            if len(bin) == 1:
                self.dist_matrix[idx] = dist_matrix[bin]
            else:
                dist_B = t.sqrt(t.square(dist_matrix[bin]).sum(dim=0)) / np.sqrt(len(bin))
                self.dist_matrix[idx] = dist_B
        self.ave_dist_bins = self.dist_matrix.mean((1, 2))
        self.dist_matrix = 1 - self.dist_matrix
        self.rel_dist_mat = self.dist_matrix
        # assert self.rel_dist_mat.min() > -1e-6 and self.rel_dist_mat.max() < 1 + 1e-6, "Relation matrix error!"
        dist_P = t.sqrt(t.square(dist_matrix).sum(dim=0)) / np.sqrt(m)
        self.rel_mat_P = 1 - dist_P

    def __make_relation_matrix_for_bins__(self, X=None):
        if X is None:
            X = self.data
        n, m = X.shape
        X = X.T.unsqueeze(-1)

        self.rel_dist_mat = t.zeros((len(self.bins),n,n), dtype=t.float32).to(device)
        for idx, bin in enumerate(self.bins):
            if len(bin) == 1:
                bin = bin[0]
                temp = X[bin]
                self.rel_dist_mat[idx] = t.cdist(temp, temp, p=1).float()
                if self.nominals[bin]:
                    self.rel_dist_mat[idx] = (self.rel_dist_mat[idx] > 1e-5).float()
            else:
                for j, bin_ in enumerate(bin):
                    temp = X[bin_]
                    mat = t.cdist(temp, temp, p=1).float()
                    if self.nominals[bin_]:
                        mat = (mat > 1e-5).float()
                    self.rel_dist_mat[idx] += t.square(mat)
                self.rel_dist_mat[idx] = t.sqrt(self.rel_dist_mat[idx])
                self.rel_dist_mat[idx] /= np.sqrt(len(bin))

        self.ave_dist_bins = self.rel_dist_mat.mean((1, 2))
        self.rel_mat_P = 1 - t.sqrt(t.square(self.rel_dist_mat).sum(dim=0)) / np.sqrt(m)
        self.rel_dist_mat = 1 - self.rel_dist_mat


    def grid_search(self, train_X, train_y, paras):
        import itertools as its
        train_X = t.from_numpy(train_X).float().to(device)
        self.__make_relation_matrix_for_bins__(train_X)

        records = dict()
        for lamb1,lamb2,lamb3 in its.combinations(paras, 3):
            self.__multi_scale_granule__([lamb1,lamb2,lamb3])
            out_scores = self.detection()
            auc = roc_auc_score(train_y, out_scores)
            pr = average_precision_score(y_true=train_y, y_score=out_scores, pos_label=1)
            records[(lamb1,lamb2,lamb3)] = auc + pr
            print('\tGrid_search: Lambs={}\tAUC={:.4f}\tPR={:.4f}\tAUC+PR={:.4f}'.format((lamb1,lamb2,lamb3), auc, pr, auc+pr))
        return list(max(records.keys(), key=lambda k:records[k]))


    def __multi_scale_granule__(self, lambs=None):
        if lambs is None:
            lambs = self.lambs
        for i in range(len(self.bins)):
            granules = self.rel_dist_mat[i].unsqueeze(0).repeat(len(lambs), 1, 1)
            cards = t.zeros((len(lambs), self.rel_dist_mat.shape[-1])).to(device)
            for idx, lamb in enumerate(lambs):
                granules[idx][granules[idx] < lamb] = 0
                cards[idx] = granules[idx].sum(dim=0)
            cards = cards/cards.sum(dim=0)
            self.rel_dist_mat[i] = (granules * cards.unsqueeze(-1)).sum(0)
            # print(self.rel_dist_mat[i][0][:10])
        # assert self.rel_dist_mat.min() > -1e-6 and self.rel_dist_mat.max() < 1 + 1e-6, "Relation matrix error!"
        # print('Multi-scale relation matrices have been built...')


    def detection(self):
        # print('Calculation Appr. Acc...')
        n_bins, n, _ = self.rel_dist_mat.shape
        weight = t.zeros((n, n_bins), dtype=t.float32).to(device)
        Acc_Appr = t.zeros((n, n_bins), dtype=t.float32).to(device)

        rel_mat_P_Negatvie = 1 - self.rel_mat_P
        for bin_idx in range(n_bins):
            unique_granules, indices = t.unique(self.rel_dist_mat[bin_idx], dim=0, return_inverse=True)
            for i, granule_i in enumerate(unique_granules):
                unique_granule_idx = t.where(indices == i)[0]

                low_appr = t.min(t.maximum(rel_mat_P_Negatvie, granule_i), dim=1).values.sum()
                up_appr = t.max(t.minimum(self.rel_mat_P, granule_i), dim=1).values.sum()

                Acc_Appr[unique_granule_idx, bin_idx] = t.clip(low_appr / up_appr, 0, 1) # Manual clipping is applied to avoid float32 precision errors
                weight[unique_granule_idx, bin_idx] = granule_i.mean()

        # print('Calculation OD degree...')
        MOF = 1 - Acc_Appr * weight
        self.MSOD = t.mean(MOF * (1 - weight**(1/3)), dim=1)
        return self.MSOD.cpu()


if __name__ == "__main__":
    pass
