import numpy as np
import matplotlib.pyplot as plt
import math
from pca import PCA

def hack_pca(filename, threshold=0.6):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)/255
    # YOUR CODE HERE
    img = img_r[:,:,0]*0.299 + img_r[:,:,1]*0.587 + img_r[:,:,2]*0.114
    H, W = img.shape
    data = []
    for i in range(H):
        # x axis
        for j in range(W):
            # y axis
            if img[i, j] >= threshold:
                data.append([i, j])
    data = np.array(data)
    N = data.shape[0]
    eigvectors, eigvalues = PCA(data)
    (vx, vy) = eigvectors[:,0]
    (vx, vy) = (vx, vy) if vy >= 0 else (-vx, -vy)
    theta = -math.asin(-vx)*180/math.pi
    R = np.array([[vy, vx], [-vx, vy]])     # rotate matrix
    odata = np.matmul(data, R)
    odata -= np.min(odata, axis=0)
    odata = odata.astype(int)
    nH, nW = np.max(odata, axis=0)
    oimg = np.zeros((nH+1, nW+1))
    for i in range(N):
        oimg[odata[i,0], odata[i,1]] = 1.
    return img, oimg, theta
    # end answer