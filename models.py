import numpy as np
from scipy.spatial.distance import cdist

class MFIOD(object):
    def __init__(self, data, nominals, lambs=None, n_bins=20):
        self.data = data
        self.nominals = nominals
        self.__find_bins__(n_bins=n_bins)
        self.__make_relation_matrix_for_bins__()
        self.__multi_scale_granule__(lambs=lambs)

    def __find_bins__(self, n_bins=20):
        data_std = self.data.std(0)
        hist, bin_edges = np.histogram(data_std, n_bins)
        bin_edges[-1] += 1e-6
        bin_indices = np.digitize(data_std, bin_edges) - 1
        self.bins = []
        for i in range(n_bins):
            indices_for_bin_i = np.where(bin_indices == i)[0]
            if len(indices_for_bin_i) > 0:
                self.bins.append(indices_for_bin_i.tolist())
        # print('Number of bins:', len(self.bins))

    def __make_relation_matrix_for_bins__(self, X=None):
        if X is None:
            X = self.data
        n, m = X.shape
        dist_matrix = np.zeros((m, n, n), dtype=np.float32)
        for i in range(m):
            col = X[:, i].reshape(n, 1)
            dist_matrix[i] = cdist(col, col, metric='cityblock').astype(np.float32)
        # assert (dist_matrix > 1 + 1e-6).sum() == 0
        if self.nominals.sum() > 0:
            dist_matrix[self.nominals] = dist_matrix[self.nominals] > 1e-6
        self.dist_matrix = np.zeros((len(self.bins),n,n), dtype=np.float32)
        for idx, bin in enumerate(self.bins):
            if len(bin) == 1:
                self.dist_matrix[idx] = dist_matrix[bin]
            else:
                dist_B = np.sqrt(np.square(dist_matrix[bin]).sum(axis=0)) / np.sqrt(len(bin))
                self.dist_matrix[idx] = dist_B
        self.ave_dist_bins = self.dist_matrix.mean((1, 2))
        self.dist_matrix = 1 - self.dist_matrix
        self.relation_matrix = self.dist_matrix
        # assert self.relation_matrix.min() > -1e-6 and self.relation_matrix.max() < 1 + 1e-6, "Relation matrix error!"

        dist_P = np.sqrt(np.square(dist_matrix).sum(axis=0)) / np.sqrt(m)
        self.R_P = 1 - dist_P
        # print('Distance matrices have been built...')

    def grid_search(self):
        pass

    def __multi_scale_granule__(self, lambs):
        if lambs == 'Default' or lambs is None:
            # Choose the three smallest ave_distance as default parameters
            lambs_new = 1 - self.ave_dist_bins
            lambs_new.sort()
            self.lambs = lambs_new[:3].round(3)
            print(self.lambs)
        elif lambs == 'Recommendation':
            exit("Grid search not implemented...")
            assert self.labels is not None, 'Grid search requires data labels'
            self.lambs = self.grid_search()
        else:
            # Assume parameters are given in a list, e.g., [0.1, 0.2, 0.3]
            # print("Using given parameters:", self.lambs)
            assert 0 < min(lambs) and 1 > max(lambs), 'Parameters not allowed...'
            self.lambs = lambs

        for i in range(len(self.bins)):
            granules = np.tile(self.relation_matrix[i][np.newaxis,:,:], (len(self.lambs),1,1))
            cards = np.zeros((len(self.lambs), self.relation_matrix.shape[-1]))
            for idx, lamb in enumerate(self.lambs):
                granules[idx][granules[idx] < lamb] = 0
                cards[idx] = granules[idx].sum(axis=0)
            cards = cards/cards.sum(axis=0)
            self.relation_matrix[i] = (granules * cards[:,:,np.newaxis]).sum(0)
            # print(self.relation_matrix[i][0][:10])
        # assert self.relation_matrix.min() > -1e-6 and self.relation_matrix.max() < 1 + 1e-6, "Relation matrix error!"
        # print('Multi-scale relation matrices have been built...')

    def detection(self):
        n, m = self.data.shape
        M = len(self.bins)
        # print('Calculation Appr. Acc...')
        weight = np.zeros((n, M), dtype=np.float32)
        Acc_Appr = np.zeros((n, M), dtype=np.float32)

        rel_mat_P = self.R_P
        rel_mat_P_N = 1 - rel_mat_P
        for l in range(M):
            rel_mat_k_l, indices = np.unique(self.relation_matrix[l], axis=0, return_inverse=True)
            for i in range(rel_mat_k_l.shape[0]):
                i_tem = np.where(indices == i)[0]
                rel_mat_B = rel_mat_k_l[i]

                low_appr = np.min(np.maximum(rel_mat_P_N, rel_mat_B), axis=1).sum()
                up_appr = np.max(np.minimum(rel_mat_P, rel_mat_B), axis=1).sum()

                Acc_Appr[i_tem, l] = low_appr / up_appr
                weight[i_tem, l] = rel_mat_k_l[i].mean()

        # print('Calculation OD degree...')
        MOF = 1 - Acc_Appr * weight
        self.MSOD = np.mean(MOF * (1 - np.power(weight, 1 / 3)), axis=1)
        return self.MSOD


if __name__ == "__main__":
    pass