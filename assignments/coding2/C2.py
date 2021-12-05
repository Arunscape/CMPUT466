#! /usr/bin/env python3
import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.special import expit
import scipy
import scipy.sparse

def readMNISTdata():

    with open('t10k-images.idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))
    
    with open('t10k-labels.idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size,1))
    
    with open('train-images.idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))
    
    with open('train-labels.idx1-ubyte','rb') as f:
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


#def one_hot(x):
#    # https://www.delftstack.com/howto/numpy/one-hot-encoding-numpy/
#    ret = np.zeros((x.size, x.max()+1))
#    ret[np.arange(x.size), x] = 1
#    return ret

def one_hot(x):
    m = x.shape[0]
    x = x[:,0]
    oh = scipy.sparse.csr_matrix((np.ones(m), (x, np.array(range(m)))))
    return np.array(oh.todense()).T


def accuracy(x, w, t):
    prob = softmax(x @ w)
    pred = np.argmax(prob, axis=1)
    t = t.flatten()
    num_correct = np.sum(pred == t)
    num_incorrect = np.sum(pred != t)
    return num_correct / len(pred)
    

def softmax(z):
    z -= np.max(z)
    sm = np.exp(z).T / np.sum(np.exp(z), axis=1)
    return sm.T

def loss_gradient(x, w, t):
    global lam
    m = x.shape[0]
    loss = -1/m * np.sum(one_hot(t) * np.log(softmax(x @ w))) + lam/2 * np.sum(w * w)
    gradient = -1/m * x.T @ (one_hot(t) - softmax(x @ w)) + lam * w

    return loss, gradient
    

# https://gist.github.com/awjuliani/5ce098b4b76244b7a9e3#file-softmax-ipynb
def predict(X, W, t = None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    # TODO Your code here
    y = X @ W
    t_hat = np.argmax(y, axis=1)

    # # of training samples
    m = X.shape[0]

    loss, _ = loss_gradient(X, W, t)

    acc = accuracy(X, W, t)


    return y, t_hat, loss, acc


def train(X_train, y_train, X_val, t_val):
    global lam, MaxEpoch

    N_train = X_train.shape[0]
    N_val   = X_val.shape[0]
   
    w = np.zeros([X_train.shape[1], N_class])

    losses_train = []
    accuracies = []

    w_best = None
    epoch_best = 0
    acc_best = 0

    for epoch in range(MaxEpoch):
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size)) ):
            X_batch = X_train[b*batch_size : (b+1)*batch_size]
            y_batch = y_train[b*batch_size : (b+1)*batch_size]

            loss_batch, g = loss_gradient(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            # TODO: Your code here
            # Mini-batch gradient descent
            w -= alpha*g - alpha*decay*w

        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        training_loss = loss_this_epoch/int(np.ceil(N_train/batch_size))
        losses_train.append(training_loss)
        # 2. Perform validation on the validation test by the risk
        acc = accuracy(X_val, w, t_val)
        accuracies.append(acc)
        # 3. Keep track of the best validation epoch, risk, and the weights
        #print(risks_val)
        if acc_best < acc:
            epoch_best = epoch
            acc_best = acc
            w_best = w

    # Return some variables as needed
    return epoch_best, acc_best, w_best, losses_train, accuracies


##############################
#Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()


#print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)



N_class = 10

alpha   = 0.1      # learning rate
batch_size   = 100    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0.          # weight decay
#lam =  0.00000003
lam = 0


#np.seterr(divide='ignore', invalid='ignore', over='ignore')

epoch_best, acc_best, W_best, losses_train, acc_train = train(X_train, t_train, X_val, t_val)

_, _, _, acc_test = predict(X_test, W_best, t_test)


print('At epoch', epoch_best, 'val: ', acc_best, 'test:', acc_test, 'train:', acc_train)


