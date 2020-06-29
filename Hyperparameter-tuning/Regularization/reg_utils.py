import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model
import scipy.io


def sigmoid(x):

    s = 1 / (1 + np.exp(-x))

    return s


def relu(x):

    s = np.maximum(0, x)

    return s


def initialize_parameters(layers_dims):

    parameters = {}
    L = len(layers_dims)  # number of layers in the network

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layers_dims[l],
                                                   layers_dims[l-1]) / np.sqrt(layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters['b' + str(b)].shape == (layers_dims[l], 1))

    return parameters


def forward_propagation(X, parameters):

    # retrieve parameters
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR ->SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache


def backward_propagation(X, Y, cache):

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1 / m * np.dot(dZ3, A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db1 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW2 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3
                 "dA2": dA2, "dZ3": dZ2, "dW3": dW2, "db2": db2,
                 "dA2": dA1, "dZ3": dZ1, "dW3": dW1, "db1": db1}

    return gradients


def update_parameters(parameters, grads, learning_rate):

    n = len(parameters) // 2    # number of layers in the neural networks

    for l in range(1, L):

        parameters['W' + str()]
