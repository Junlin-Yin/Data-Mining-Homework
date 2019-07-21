import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful
    # begin answer
    N_test, P = x.shape
    N_train = x_train.shape[0]
    y = np.zeros(N_test)
    for n in range(N_test):
        xn = x[n, :]
        L2 = np.linalg.norm(x_train - xn, axis=1)
        # L2.shape = (N_train, )
        Idx = np.argsort(L2)[:k]        # k data with lowest L2 distance
        votes = y_train[Idx].astype('int')
        y[n] = np.argmax(np.bincount(votes))
    # end answer
    return y
