import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import cdist

def MFIOD_example(data, lambs):
    n, m = data.shape
    n_bins = 2
    data_std = data.std(0)
    print('Standard deviation for each attribute:',data_std)
    hist, bin_edges = np.histogram(data_std, n_bins)
    bin_edges[-1] += 1e-6
    bin_indices = np.digitize(data_std, bin_edges) - 1
    bins = []
    for i in range(n_bins):
        indices_for_bin_i = np.where(bin_indices == i)[0]
        if len(indices_for_bin_i) > 0:
            bins.append(indices_for_bin_i.tolist())
    print('Attribute bins:', bins)

    n_lamb = len(lambs)
    dist_matrix = np.zeros((m, n, n), dtype=np.float32)
    for i in range(m):
        col = data[:, i].reshape(n, 1)
        dist_matrix[i] = cdist(col, col, metric='cityblock').astype(np.float32)
    dist_B = np.zeros((len(bins), n, n), dtype=np.float32)
    for idx, bin in enumerate(bins):
        if len(bin) == 1:
            dist_B[idx] = dist_matrix[bin]
        else:
            dist_B[idx] = np.sqrt(np.square(dist_matrix[bin]).sum(axis=0)) / np.sqrt(len(bin))
    rel_mat_k = 1 - dist_B
    dist_P = np.sqrt(np.square(dist_matrix).sum(axis=0)) / np.sqrt(m)
    rel_mat_P = 1 - dist_P
    print('Relation matrix for attribute set P:\n============================\n', rel_mat_P, '\n============================')
    # np.savetxt('example_M_R_P.csv', rel_mat_P, fmt='%.3f', delimiter=',')

    for i in range(len(bins)):
        granules = np.tile(rel_mat_k[i][np.newaxis, :, :], (n_lamb, 1, 1))
        cards = np.zeros((n_lamb, rel_mat_k.shape[-1]))
        for idx, lamb in enumerate(lambs):
            granules[idx][granules[idx] < 1-lamb] = 0
            # print(granules[idx])
            # np.savetxt('example_B{}_granule{}.csv'.format(i,lamb), granules[idx],fmt='%.6f', delimiter=',')
            cards[idx] = granules[idx].sum(axis=0)
        cards = cards / cards.sum(axis=0)
        rel_mat_k[i] = (granules * cards[:, :, np.newaxis]).sum(0)
        # print(rel_mat_k[i])
        # np.savetxt('example_multi_granule_B{}.csv'.format(i), rel_mat_k[i], fmt='%.6f',delimiter=',')
    print('Relation matrices for attribute bins:\n',rel_mat_k)

    M =n_bins
    weight = np.zeros((n, M), dtype=np.float32)
    Acc_Appr = np.zeros((n, M), dtype=np.float32)

    rel_mat_P_N = 1 - rel_mat_P
    for l in range(M):
        rel_mat_k_l, indices = np.unique(rel_mat_k[l], axis=0, return_inverse=True)
        for i in range(rel_mat_k_l.shape[0]):
            i_tem = np.where(indices == i)[0]
            rel_mat_B = rel_mat_k_l[i]

            low_appr = np.min(np.maximum(rel_mat_P_N, rel_mat_B), axis=1)
            # print(l,i_tem,low_appr)
            low_appr = low_appr.sum()
            up_appr = np.max(np.minimum(rel_mat_P, rel_mat_B), axis=1)
            # print(l,i_tem,up_appr)
            up_appr = up_appr.sum()

            Acc_Appr[i_tem, l] = low_appr / up_appr
            # print(l,i_tem,low_appr / up_appr)
            weight[i_tem, l] = rel_mat_k_l[i].mean()
            # print(weight[i_tem, l])

    # print('Calculation OD degree...')
    MOF = 1 - Acc_Appr * weight
    print('MOF:\n', MOF)
    MSOD = np.mean(MOF * (1 - np.power(weight, 1 / 3)), axis=1)
    print(1 - np.power(weight, 1 / 3))
    print('MSOD:\n', MSOD)

if __name__ == "__main__":

    data = np.array(
           [[0.7, 9, 1],
            [0.3, 6, 0 ],
            [0.5, 2, 1 ],
            [0.2, 3, 0 ],
            [0.4, 7, 1 ],
            [0.6, 3, 0 ]])

    data = minmax_scale(data)
    print('Data after mim-max scaling:\n', data)

    MFIOD_example(data, lambs=[0.2, 0.5])
