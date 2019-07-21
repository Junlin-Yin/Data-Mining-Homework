import numpy as np

def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    Xb = np.vstack((np.ones((1, X.shape[1])), X))
    iters = 0
    MAX_ITERS = 1e3
    EPSILON = 1e-1
    LEARNING_RATE = 1e-1
    while(iters < MAX_ITERS):
        # calculate y_hat
        y_hat = 1/(1+np.exp(-np.matmul(w.T, Xb)))
        # calculate gradient
        grad = np.matmul(-(y-y_hat), Xb.T).T
        if(np.linalg.norm(grad) < EPSILON):
            break
        w = w - LEARNING_RATE*grad
    # end answer
    
    return w
