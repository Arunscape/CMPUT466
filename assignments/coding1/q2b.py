#!/usr/bin/env python3

import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import math

def predict(X, w, y=None, denormalize=False):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample
    global std_y, mean_y
    # TODO: Your code here
    y_hat = X.dot(w)
    if denormalize:
        y_hat = y_hat * std_y + mean_y
        y = y * std_y + mean_y

    loss  = 1 / (2*len(y)) * (y_hat - y).T.dot(y_hat - y)
    risk  = 1/ len(y) * np.sum(np.abs(y_hat-y))
    
    return y_hat, loss.item(0), risk

def train(X_train, y_train, X_val, y_val, decay):
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
            w = w - alpha * ( 1/ len(y_batch) * (X_batch.T).dot(y_hat_batch-y_batch) + decay * w)

        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        training_loss = loss_this_epoch/batch_size
        losses_train.append(training_loss)
        # 2. Perform validation on the validation test by the risk
        _, _, risk = predict(X_val, w, y_val, denormalize=True)
        risks_val.append(risk)
        # 3. Keep track of the best validation epoch, risk, and the weights
        #print(risks_val)
        if risk == np.min(risks_val):
            epoch_best = epoch
            risk_best = risk
            w_best = w

    # Return some variables as needed
    return epoch_best, risk_best, w_best, risks_val, losses_train


def tune_hyperparameter(X_train, y_train, X_val, y_val, hyperparameters=(3, 1, 0.3, 0.1, 0.03, 0.01)):
    lowest_risk = math.inf
    best_param = None
    epoch = None
    risk = None
    risk_plot = None
    losses_plot = None
    w_best = None
    for h in hyperparameters:
        best_epoch, best_risk, w, risks_val, losses_train = train(X_train, y_train, X_val, y_val, h)
        lowest_risk = min(lowest_risk, best_risk)
        if best_risk == lowest_risk:
            best_param = h
            epoch = best_epoch
            risk = best_risk
            risk_plot = risks_val
            losses_plot = losses_train
            w_best = w

    return best_param, epoch, risk, w_best, risk_plot, losses_plot

        
############################
# Main code starts here
############################

# Load data. This is the only allowed API call from sklearn
X, y = datasets.load_boston(return_X_y=True)
y = y.reshape([-1, 1])
# X: sample x dimension
# y: sample x 1

mean_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0)
X = (X - mean_X) / std_X

X_extended = np.concatenate((X, np.square(X)), axis=1)

# Augment feature
X_ = np.concatenate( ( np.ones([X_extended.shape[0],1]), X_extended ), axis=1)
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
best_param, best_epoch, best_risk, w, risks_val, losses_train = tune_hyperparameter(X_train, y_train, X_val, y_val)
y_hat, loss, test_risk = predict(X_test, w, y_test, denormalize=True)
print(f"""2 b)
best hyperparameter: {best_param}
1. Best epoch: {best_epoch}
2. Validation performance: {best_risk}
3. Test performance: {test_risk}""")


#print(risks_val)
#print(losses_train)
plt.plot(range(MaxIter), losses_train)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("images/q2b_loss.png")

plt.figure()

plt.plot(range(MaxIter), risks_val)
plt.xlabel("Epoch")
plt.ylabel("Risk")
plt.savefig("images/q2b_risk.png")

# Perform test by the weights yielding the best validation performance

# Report numbers and draw plots as required. 
