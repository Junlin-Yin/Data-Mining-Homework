import numpy as np
import extract_image
import knn

def hack(img_name):
    '''
    HACK Recognize a CAPTCHA image
      Inputs:
          img_name: filename of image
      Outputs:
          digits: 1x5 matrix, 5 digits in the input CAPTCHA image.
    '''
    data = np.load('hack_data.npz')

    # YOUR CODE HERE (you can delete the following code as you wish)
    x_train = data['x_train']   # x_train.shape = (50, 140)
    y_train = data['y_train']   # y_train.shape = (50, )
    # begin answer
    x_test = extract_image.extract_image(img_name)  # x_test.shape = (5, 140)
    k = 10
    digits = knn.knn(x_test, x_train, y_train, k)   # digits.shape = (5, )
    # end answer

    return digits