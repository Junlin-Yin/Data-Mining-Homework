import numpy as np

def ridge(X, y, lmbda):
    '''
    RIDGE Ridge Regression.

      INPUT:  X: training sample features, P-by-N matrix.
              y: training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w: learned parameters, (P+1)-by-1 column vector.

    NOTE: You can use pinv() if the matrix is singular.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    Xb = np.vstack((np.ones((1, X.shape[1])), X))
    tmp = np.matmul(Xb, Xb.T) + lmbda * np.eye(P+1, dtype='double')
    coe = np.matmul(np.linalg.pinv(tmp), Xb)
    w = np.matmul(coe, y.T)
    # end answer
    return w
