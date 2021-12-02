# -*- coding: utf-8 -*-


import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.special import expit

def readMNISTdata():

    with open('t10k-images-idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))
    
    with open('t10k-labels-idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size,1))
    
    with open('train-images-idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))
    
    with open('train-labels-idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size,1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate( ( np.ones([train_data.shape[0],1]), train_data ), axis=1)
    test_data  = np.concatenate( ( np.ones([test_data.shape[0],1]),  test_data ), axis=1)
    np.random.seed(314)
    np.random.shuffle(train_labels)
    np.random.seed(314)
    np.random.shuffle(train_data)

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val   = train_data[50000:] /256
    t_val   = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data, test_labels




def predict(X, W, t = None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    # TODO Your code here

    return y, t_hat, loss, acc



def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val   = X_val.shape[0]
   
    #TODO Your code here

    return epoch_best, acc_best,  W_best


##############################
#Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()


print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)



N_class = 10

alpha   = 0.1      # learning rate
batch_size   = 100    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0.          # weight decay


epoch_best, acc_best, W_best = train(X_train, t_train, X_val, t_val)

_, _, _, acc_test = predict(X_test, W_best, t_test)


print('At epoch', epoch_best, 'val: ', acc_best, 'test:', acc_test, 'train:', acc_train)


