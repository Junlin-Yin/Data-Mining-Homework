import numpy as np


def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    # YOUR CODE HERE

    # begin answer
    N, P = x.shape
    idx = np.zeros(N, dtype='int') - 1
    ctrs_idx = np.array(list(range(N)))
    np.random.shuffle(ctrs_idx)
    ctrs_idx = ctrs_idx[:k]
    ctrs = x[ctrs_idx, :]      # shape: (k, P)
    iter_ctrs = np.zeros((1, k, P))
    iter_ctrs[0, :, :] = ctrs
    while True:
        # every points find its class
        new_idx = np.zeros(N, dtype='int')
        for n in range(N):
            xn = x[n, :]
            L2 = np.linalg.norm(ctrs - xn, axis=1)
            new_idx[n] = np.argmin(L2)  # xn belongs to class new_idx[n]
        if(np.sum(idx != new_idx) == 0):
            # no change, break
            break
        idx = np.copy(new_idx)
        # update ctrs
        for i in range(k):
            tmp = np.argwhere(idx == i).squeeze()
            xn = x[tmp, :]
            if(np.sum(idx == i) == 0):
                print('wrong')
            ctrs[i, :] = np.sum(xn, axis=0)/np.sum(idx == i)
        ctrs_newaxis = ctrs[np.newaxis, :, :]
        iter_ctrs = np.concatenate((iter_ctrs, ctrs_newaxis), axis=0)
    # end answer
    return idx, ctrs, iter_ctrs
