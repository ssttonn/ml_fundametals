import numpy as np
import scipy
import matplotlib.pyplot as plt

def load_2D_dataset():
    data = scipy.io.loadmat('../../dataset/data.mat', verify_compressed_data_integrity=False)
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);
    
    return train_X, train_Y, test_X, test_Y