import numpy as np
import math

def gaussian_value(x, mu, sigma):
    # 2*pi is omitted
    det_sigma = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    exponent = -0.5*np.dot(np.transpose(x-mu), np.dot(sigma_inv, x-mu))
    return math.exp(exponent)/math.sqrt(det_sigma)

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    p = np.zeros((N, K))
    #Your code HERE

    # begin answer
    for i in range(N):
        # calculate P(x_i)
        P_xi = 0
        for j in range(K):
            # consider p[i][j]
            p[i][j] = gaussian_value(X[:,i], Mu[:,j], Sigma[:,:,j])*Phi[j]
            P_xi += p[i][j]
        # regularization
        p[i] = p[i] / P_xi
    # end answer
    
    return p
    