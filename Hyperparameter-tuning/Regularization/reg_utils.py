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
