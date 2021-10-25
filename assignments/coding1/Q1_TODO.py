#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def rotate(data, degree):
    # data: M x 2
    theta = np.pi/180 * degree
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]] )# rotation matrix
    return np.dot(data, R.T)


def leastSquares(X, Y):
    # In this function, X is always the input, Y is always the output
    # X: M x (d+1), Y: M x 1, where d=1 here
    # return weights w
    
    # TODO: YOUR CODE HERE
    # closed form solution by matrix-vector representations only

    return w


def model(X, w):
    # X: M x (d+1)
    # w: d+1
    # return y_hat: M x 1

    # TODO: YOUR CODE HERE

    return y_hat


def generate_data(M, var1, var2, degree):

    # data generate involves two steps:
    # Step I: generating 2-D data, where two axis are independent
    # M (scalar): The number of data samples
    # var1 (scalar): variance of a
    # var2 (scalar): variance of b
    
    mu = [0, 0]

    Cov = [[var1, 0],
           [0,  var2]]

    data = np.random.multivariate_normal(mu, Cov, M)
    # shape: M x 2

    plt.figure()
    plt.scatter(data[:,0], data[:, 1], color="blue")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.tight_layout()
    plt.savefig('data_ab_'+str(var2)+'.jpg')


    # Step II: rotate data by 45 degree counter-clockwise,
    # so that the two dimensions are in fact correlated


    data = rotate(data, degree)
    plt.tight_layout()
    plt.figure()
    # plot the data points
    plt.scatter(data[:,0], data[:, 1], color="blue")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('x')
    plt.ylabel('y')

    # plot the line where data are mostly generated around
    X_new = np.linspace(-5, 5, 100, endpoint=True).reshape([100,1])

    Y_new = np.tan(np.pi/180*degree)*X_new
    plt.plot(X_new, Y_new, color="blue", linestyle='dashed')
    plt.tight_layout()
    plt.savefig('data_xy_'+str(var2)+ '_' + str(degree) + '.jpg')
    return data

###########################
# Main code starts here
###########################
# Settings
M = 5000
var1 = 1
var2 = 0.8
degree = 45

data = generate_data(M, var1, var2, degree)

##########
# Training the linear regression model predicting y from x (x2y)
Input  = data[:,0].reshape((-1,1)) # M x d, where d=1
Input_aug = np.concatenate([Input, np.ones([M, 1])], axis=1) # M x (d+1) augmented feature
Output = data[:,1].reshape((-1,1)) # M x 1


w_x2y = leastSquares(Input_aug, Output) # (d+1) x 1, where d=1

print('Predicting y from x (x2y): weight='+ str(w_x2y[0,0]), 'bias = ', str(w_x2y[1,0]))

# # Training the linear regression model predicting x from y (y2x)

Input  = data[:,1].reshape((-1,1)) # M x d, where d=1
Input_aug = np.concatenate([Input, np.ones([M, 1])], axis=1) # M x (d+1) augmented feature
Output = data[:,0].reshape((-1,1)) # M x 1

w_y2x = leastSquares(Input_aug, Output) # (d+1) x 1, where d=1
print('Predicting x from y (y2x): weight='+ str(w_y2x[0,0]), 'bias = ', str(w_y2x[1,0]))


# plot the data points
plt.figure()
X = data[:,0].reshape((-1,1)) # M x d, where d=1
Y = data[:,1].reshape((-1,1)) # M x d, where d=1

plt.scatter(X, Y, color="blue", marker='x')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel('x')
plt.ylabel('y')

# plot the line where data are mostly generated around
X_new = np.linspace(-4, 4, 100, endpoint=True).reshape([100, 1])

Y_new = np.tan(np.pi/180*degree)*X_new
plt.plot(X_new, Y_new, color="blue", linestyle='dashed')

# plot the prediction of y from x (x2y)
X_new = np.linspace(-4, 4, 100, endpoint=True).reshape([100, 1]) # M x d, where d=1
X_new_aug = np.concatenate([X_new, np.ones([X_new.shape[0], 1])], axis=1) # M x (d+1) augmented feature
plt.plot(X_new, model(X_new_aug, w_x2y), color="red", label="x2y")

# plot the prediction of x from y (y2x)
Y_new = np.linspace(-4, 4, 100, endpoint=True).reshape([100, 1]) # M x d, where d=1
Y_new_aug = np.concatenate([X_new, np.ones([X_new.shape[0], 1])], axis=1) # M x (d+1) augmented feature
plt.plot(model(Y_new_aug, w_y2x), Y_new, color="green", label="y2x")
plt.legend()
plt.tight_layout()
plt.savefig('Regression_model_' + str(var2) + '_' + str(degree) + '.jpg')
