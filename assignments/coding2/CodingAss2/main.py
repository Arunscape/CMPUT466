#!/usr/bin/env python3

from utils import plot_data, generate_data
import numpy as np
"""
Documentation:

Function generate() takes as input "A" or "B", it returns X, t.
X is two dimensional vectors, t is the list of labels (0 or 1).    

Function plot_data(X, t, w=None, bias=None, is_logistic=False, figure_name=None)
takes as input paris of (X, t) , parameter w, and bias. 
If you are plotting the decision boundary for a logistic classifier, set "is_logistic" as True
"figure_name" specifies the name of the saved diagram.
"""

NUM_ITER = 1000000
alpha = 0.01

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, t):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    """
    # initialize w
    n, m = X.shape
    w = np.zeros(m)
    b = 0
    for _ in range(NUM_ITER):
        y_hat = sigmoid(X @ w  + b)
        w -= alpha * X.T @ (y_hat-t) / m
        b -= alpha * np.sum(y_hat-t) / m

    return w, b

def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """

    # 1 if sigmoid(z) >= 0.5 else 0
    t = sigmoid(X @ w + b) >= 0.5
    return t * 1

def train_linear_regression(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """
    n, m = X.shape
    
    w = np.zeros(m)
    b = 0


    X = np.append(X, np.ones((n,1)), axis=1) 

    w = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(t)

    b = w[-1]
    w = w[:-1]
    return w, b

def predict_linear_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    t = (X @ w + b) >= 0
    return t * 1

def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """

    # number correctly predicted / number total samples
    acc = np.sum(t == t_hat) / len(t)
    return acc

def main():
    # Dataset A
    # Linear regression classifier
    X, t = generate_data("A")
    print(X.shape, t.shape)
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False, figure_name='dataset_A_linear.png')

    # logistic regression classifier
    X, t = generate_data("A")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True, figure_name='dataset_A_logistic.png')

    # Dataset B
    # Linear regression classifier
    X, t = generate_data("B")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False, figure_name='dataset_B_linear.png')

    # logistic regression classifier
    X, t = generate_data("B")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True, figure_name='dataset_B_logistic.png')

main()
