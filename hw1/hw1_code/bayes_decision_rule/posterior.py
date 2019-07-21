import numpy as np
from likelihood import likelihood

def posterior(x):
    '''
    POSTERIOR Two Class Posterior Using Bayes Formula
    INPUT:  x, features of different class, C-By-N vector
            C is the number of classes, N is the number of different feature
    OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
    '''

    C, N = x.shape
    l = likelihood(x)
    total = np.sum(x)
    prior = np.sum(x, axis=1) / total
    p = np.zeros((C, N))
    #TODO

    # begin answer
    for c in range(C):
        p[c] = l[c] * prior[c] / (l[0] * prior[0] + l[1] * prior[1])
    # end answer
    
    return p
