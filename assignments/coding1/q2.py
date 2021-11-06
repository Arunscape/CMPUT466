#!/usr/bin/env python3

import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt


def predict(X, w, y = None):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample

    # TODO: Your code here
    y_hat = X.dot(w)
    loss  = 1 / (2*len(y)) * (y_hat - y).T.dot(y_hat - y)
    risk  = 1/ len(y) * np.sum(np.abs(y_hat-y))
    
    return y_hat, loss.flatten(), risk


def train(X_train, y_train, X_val, y_val):
    N_train = X_train.shape[0] # number of samples
    N_val   = X_val.shape[0]   # number of validation samples

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val   = []

    w_best    = None
    risk_best = 10000
    epoch_best= 0
    
    for epoch in range(MaxIter):

        loss_this_epoch = 0
        for b in range( int(np.ceil(N_train/batch_size)) ):
            
            X_batch = X_train[b*batch_size : (b+1)*batch_size]
            y_batch = y_train[b*batch_size : (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            # TODO: Your code here
            # Mini-batch gradient descent
            w = w - 1/ len(y_batch) * alpha * (X_batch.T).dot(y_hat_batch-y_batch)

        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        training_loss = loss_this_epoch/batch_size
        losses_train.append(training_loss)
        # 2. Perform validation on the validation test by the risk
        y_hat, _, risk = predict(X_val, w, y_val)
        risks_val.append(risk)
        # 3. Keep track of the best validation epoch, risk, and the weights
        #print(risks_val)
        if risk == np.min(risks_val):
            epoch_best = epoch
            risk_best = risk
            w_best = w

    # Return some variables as needed
    return epoch_best, risk_best, w_best, risks_val, losses_train



############################
# Main code starts here
############################

# Load data. This is the only allowed API call from sklearn
X, y = datasets.load_boston(return_X_y=True)
y = y.reshape([-1, 1])
# X: sample x dimension
# y: sample x 1

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)



# Augment feature
X_ = np.concatenate( ( np.ones([X.shape[0],1]), X ), axis=1)
# X_: Nsample x (d+1)

# normalize features:
mean_y = np.mean(y)
std_y  = np.std(y)

y = (y - np.mean(y)) / np.std(y)

#print(X.shape, y.shape) # It's always helpful to print the shape of a variable


# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_[:300]
y_train = y[:300]

X_val   = X_[300:400]
y_val   = y[300:400]

X_test = X_[400:]
y_test = y[400:]

#####################
# setting

alpha   = 0.001      # learning rate
batch_size   = 10    # batch size
MaxIter = 100        # Maximum iteration
decay = 0.0          # weight decay




# TODO: Your code here
best_epoch, best_risk, w, risks_val, losses_train = train(X_train, y_train, X_val, y_val)
y_hat, loss, test_risk = predict(X_test, w, y_test)
print(f"2 a)\n1. Best epoch: {best_epoch}\n2. Validation performance: {best_risk}\n3. Test performance: {test_risk}")
print(f"loss: {loss}")


print(risks_val)
print(losses_train)
plt.plot(range(MaxIter), losses_train)
plt.show()

plt.plot(range(MaxIter), risks_val)
plt.show()

# Perform test by the weights yielding the best validation performance

# Report numbers and draw plots as required. 
