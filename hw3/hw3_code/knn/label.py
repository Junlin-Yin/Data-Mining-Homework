from hack import hack
from extract_image import extract_image
from show_image import show_image
import scipy.misc
import numpy as np
N_train = 100
y = []
for i in range(N_train):
    path = './checkcode/' + str(i) + '.aspx'
    m = scipy.misc.imread(path, mode='L')
    d1 = m[4:18, 4:14].reshape(140)
    d2 = m[4:18, 13:23].reshape(140)
    d3 = m[4:18, 22:32].reshape(140)
    d4 = m[4:18, 31:41].reshape(140)
    d5 = m[4:18, 40:50].reshape(140)
    d = np.vstack((d1, d2, d3, d4, d5))     # d.shape = (5, 140)
    show_image(d)
    if(i == 0):
        x = np.copy(d)
    else:
        x = np.vstack((x, d))
    label = list(input("Enter 5 labels here: "))
    y.append(label)
y = np.array(y)
y = y.reshape(y.shape[0]*y.shape[1])
print(x.shape, y.shape)
np.savez('hack_data.npz', x_train = x, y_train = y)
print('save done')