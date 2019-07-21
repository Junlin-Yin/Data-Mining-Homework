import numpy as np
from kmeans import kmeans
import matplotlib.pyplot as plt

def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters
        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    tmp = np.sum(W, axis=1)
    D = np.diag(tmp)
    D_m05 = np.diag(tmp**-0.5)    # D^(-0.5)
    M = np.matmul(np.matmul(D_m05, D-W), D_m05)
    eigvals, eigvecs = np.linalg.eig(M)
    Dict = zip(eigvals, range(len(eigvals)))
    sDict = sorted(Dict, key=lambda x:x[0])
    Z = np.vstack([eigvecs[:,i] for (v, i) in sDict[:k]]).T     # Z.shape = (N, k)
    idx = kmeans(Z, k)
    return idx
    # end answer
