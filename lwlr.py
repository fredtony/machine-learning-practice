import numpy as np
import matplotlib.pylab as plt

def lwlrClassifier(X_train, y_train, x, tau):
    '''
    Locally weighter linear regression
    Takes in a training set with a new input x (numpy array) and returns a
    predicted y along with the associated theta values. tau is a parameter
    of the algorithm which impacts how quickly 	weights for each point in the
    training set fall off as they get further from x.
    '''
    var = .0001
    m,n = X_train.shape
    weighted_X = x - X_train
    w = np.exp(-(np.linalg.norm(weighted_X, axis=1) / float(2 * tau ** 2)))
    theta = np.random.normal(size=(n,), )
    z = w * (y_train - h(theta, X_train))
    grad = np.matmul(X_train.T,z) - var * theta
    iter = 0
    while np.linalg.norm(grad) > .000001:
        iter += 1
        if iter > 30:
            print "ERROR: Not converging to solution"
        h_theta = h(theta, X_train)
        D = (-1 * w * h_theta * (1 - h_theta)) * np.eye(m)
        hess = np.matmul(np.matmul(X_train.T,D),X_train) - var * np.eye(n)
        z = w * (y_train - h_theta)
        grad = np.matmul(X_train.T,z) - var * theta
        theta = theta - np.matmul(np.linalg.inv(hess), grad)
    y = (1+np.exp(theta.dot(x)))**-1
    if y > 0.5:
        return 1
    else:
        return 0

#def plot_lwlr(X_train, y_train, tau, resolution)

def h(theta, X_train):
	return (1 + np.exp(np.matmul(-1*X_train,theta))) ** -1

#x_train = 
#y_train = 

