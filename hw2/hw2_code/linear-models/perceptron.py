import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    MAX_ITERS = 1e4
    # YOUR CODE HERE
    
    # begin answer
    while(iters < MAX_ITERS):
        done = True
        for i in range(N):
                # for every sample
                iters += 1
                xi = np.hstack((np.ones((1)), X[:,i]))
                xi = xi.reshape(xi.shape[0], 1)
                tmp = np.matmul(w.T, xi)
                yi_hat = np.sign(tmp)
                if(yi_hat[0][0] != y[0][i]):
                        done = False
                        w = w + xi*y[0][i]
        if(done == True):
                # ok
                break
    # end answer
    
    return w, iters