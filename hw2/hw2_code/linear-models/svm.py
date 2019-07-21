import numpy as np
from scipy.optimize import minimize
def fun(args):
    return lambda w: np.dot(w, w)

def con(args):
    Xb, y= args
    return ({'type': 'ineq', 'fun': lambda w: y[0, :]*np.matmul(w.T, Xb) - 1})

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.ones((P+1, 1))
    num = 0

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    RATE = 10
    EPSILON = 1e-2
    Xb = np.vstack((np.ones((1, X.shape[1])), X))
    cons = con((Xb, y))
    w = minimize(fun(None), w, method='SLSQP', constraints=cons).x
    y_hat = np.matmul(Xb.T, w).T
    num = N - np.sum(y*y_hat < 1-EPSILON) - np.sum(y*y_hat > 1+EPSILON)
    # end answer
    return w, num

