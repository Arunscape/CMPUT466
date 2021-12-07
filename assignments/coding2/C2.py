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

#def one_hot(x):
#    m = x.shape[0]
#    x = x[:,0]
#    oh = scipy.sparse.csr_matrix((np.ones(m), (x, np.array(range(m)))))
#    return np.array(oh.todense()).T


def one_hot(y):
    global N_class
    y_hot = np.zeros((len(y), N_class))
    y_hot[np.arange(len(y)), y] = 1

    return y_hot



def accuracy(y, y_hat):
    return np.sum(y==y_hat)/len(y)
    

def softmax(z):
    #exp = np.exp(z-np.max(z))
    #for i in range(len(z)):
    #    exp[i] /= np.sum(exp[i])
    #return exp
    z -= np.max(z)
    sm = np.exp(z).T / np.sum(np.exp(z), axis=1)
    return sm.T

#def loss_gradient(x, w, t):
#    global lam
#    m = x.shape[0]
#    loss = -1/m * np.sum(one_hot(t) * np.log(softmax(x @ w))) + lam/2 * np.sum(w * w)
#    gradient = -1/m * x.T @ (one_hot(t) - softmax(x @ w)) + lam * w
#
#    return loss, gradient

def get_loss(y, y_hat):
    return -np.mean(np.log(y_hat[np.arange(len(y)), y]))
    

# https://gist.github.com/awjuliani/5ce098b4b76244b7a9e3#file-softmax-ipynb
def predict(X, w, t = None, debug=False):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    # TODO Your code here
    y = X @ w
    y_hat = softmax(y)

    if debug:
        print("y: ", y)
        print("y_hat: ", y_hat)

    t_hat = np.argmax(y_hat, axis=1)

    # # of training samples

    loss = get_loss(t, y_hat)
    acc = accuracy(t.flatten(), t_hat)

    if debug:
        print("t: ", t.flatten())
        print("t_hat: ", t_hat)


    return y, t_hat, loss, acc

# https://towardsdatascience.com/softmax-regression-in-python-multi-class-classification-3cb560d90cb2
def train(X_train, y_train, X_val, t_val):
    global MaxEpoch
   
    m, n = X_train.shape

    # add column for bias
    w = np.zeros((n, N_class))

    losses_train = []
    accuracies = []

    w_best = None
    epoch_best = 0
    acc_best = 0
    
    N_train = X_train.shape[0] # number of samples
    N_val   = X_val.shape[0]   # number of validation samples

    for epoch in range(MaxEpoch):
        loss_this_epoch = 0
        for i in range(int(np.ceil(N_train/batch_size)) ):
        
            X_batch = X_train[i*batch_size : (i+1)*batch_size]
            y_batch = y_train[i*batch_size : (i+1)*batch_size]

            y, _, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            y_hat = softmax(y)
            y_hot = one_hot(y_batch)

            w -= alpha / m * X_batch.T @ (y_hat - y_hot)

        training_loss = loss_this_epoch/int(np.ceil(N_train/batch_size))
        losses_train.append(training_loss)
        _, _, _, acc = predict(X_val, w, t_val, debug=True)
        accuracies.append(acc)

        print("loss: ", training_loss)
        print("accuracy: ", acc)

        if acc_best < acc:
            epoch_best = epoch
            acc_best = acc
            w_best = w

    # Return some variables as needed
    return epoch_best, acc_best, w_best, losses_train, accuracies


num_batches = 50
def plot_training_losses(losses_train):
    fig = plt.figure()
    plt.plot(losses_train, label="Training Losses")
    plt.title("Training Losses\n{}".format(f"alpha={alpha} num_batches={num_batches} batch_size={batch_size} MaxEpoch={MaxEpoch} decay={decay} lam={lam}"))
    plt.legend()
    plt.xlabel('Number of epoch')
    plt.ylabel('Training Loss')
    plt.show()
    
    
def plot_training_accuracy(acc_val):
    fig = plt.figure()
    plt.plot(acc_val, label="Validation Accuracy")
    plt.title("Validation Accuracy\n{}".format(f"alpha={alpha} num_batches={num_batches} batch_size={batch_size} MaxEpoch={MaxEpoch} decay={decay} lam={lam}"))
    plt.legend()
    plt.xlabel('Number of epoch')
    plt.ylabel('Validation Accuracy')
    plt.show()


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

print("w: ", W_best)

_, _, _, acc_test = predict(X_test, W_best, t_test)


print('At epoch', epoch_best, 'val: ', acc_best, 'test:', acc_test, 'train:', acc_train)

plot_training_accuracy(acc_train)
plot_training_losses(losses_train)

