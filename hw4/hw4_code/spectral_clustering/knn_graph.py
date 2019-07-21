import numpy as np

def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold
        Output:
            W - adjacency matrix, n-by-n matrix.
    '''

    # YOUR CODE HERE
    # begin answer
    N, P = X.shape
    sigma = 0.1
    W = np.zeros((N, N))
    for i in range(N):
        L2 = np.linalg.norm(X - X[i, :], axis=1)
        Idx = np.argsort(L2)[1:k+1]
        for j in Idx:
            similarity = np.exp(-np.sum(np.square(X[i, :] - X[j, :]))/(2*sigma**2))
            W[i, j] = W[j, i] = similarity if similarity > threshold else 0
    return W
    # end answer
