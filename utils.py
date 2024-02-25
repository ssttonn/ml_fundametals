import numpy as np
import time

import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    ds = s * (1 - s)
    return s, ds


def tanh(Z):
    t = np.tanh(Z)
    dt = 1 - t**2
    return t, dt


def leaky_relu(Z):
    r = np.maximum(Z * 0.01, Z)
    dr = np.where(Z > 0, 1, 0.01)
    return r, dr


def relu(Z):
    r = np.maximum(np.zeros(Z.shape), Z)
    dr = np.where(Z > 0, 1, 0)
    return r, dr


def linear(Z):
    return Z, 1


def input_output_layer_sizes(X, y):
    return X.shape[0], y.shape[0]


def randomly_initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def zeros_initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def xavier_initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def he_initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def forward_propagation(X, parameters, activations, keep_prob=1):
    assert(keep_prob > 0 and keep_prob <= 1)
    cache = {}

    new_parameters = parameters.copy()

    # Forward propagation
    for i, activation in enumerate(activations):
        l = i + 1
        A_last = X if l == 1 else cache[f"A{l-1}"]
        W = new_parameters[f"W{l}"]
        b = new_parameters[f"b{l}"]
        Z = np.dot(W, A_last) + b
        A = activation(Z)[0]
        D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
        if l < len(activations):
            A = A * D
            A = A / keep_prob
            cache[f"D{l}"] = D
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A
        

    return cache


def compute_sigmoid_cost(A_last, Y):
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y, np.log(A_last).T) - np.dot(1-Y, np.log(1-A_last).T))
    cost = np.squeeze(cost)
    return cost


def compute_sigmoid_cost_with_regularization(A_last, Y, parameters, layer_dims, lamd):
    m = Y.shape[1]
    cost = compute_sigmoid_cost(A_last, Y)
    if lamd == 0:
        return cost

    L2_regularization_cost = 0
    for i in range(1, len(layer_dims)):
        L2_regularization_cost += np.sum(np.square(parameters[f"W{i}"]))
    return cost + (lamd / (2 * m)) * L2_regularization_cost


def backward_propagation(X, Y, parameters, cache, activations, lamd, keep_prob):
    cache = cache.copy()
    X = X.copy()
    grads = {}
    dZs = {}
    m = X.shape[1]

    # Backward propagation
    for i, activation in reversed(list(enumerate(activations))):
        current_layer_number = i + 1

        A = cache[f"A{current_layer_number}"]
        Z = cache[f"Z{current_layer_number}"]
        if current_layer_number == len(activations):
            dZ = (A - Y)
        else:
            W_last = parameters[f"W{current_layer_number+1}"]
            dZ_last = dZs[f"dZ{current_layer_number+1}"]
            dA = np.dot(W_last.T, dZ_last)
            D = cache[f"D{current_layer_number}"]
            dA = dA * D
            dA = dA / keep_prob
            dZ = dA * activation(Z)[1]

        if current_layer_number == 1:
            A_next = X
        else:
            A_next = cache[f"A{current_layer_number - 1}"]
        
        dW = (1 / m) * np.dot(dZ, A_next.T) + (lamd / m) * parameters[f"W{current_layer_number}"]
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dZs[f"dZ{current_layer_number}"] = dZ
        grads[f"dW{current_layer_number}"] = dW
        grads[f"db{current_layer_number}"] = db

    return grads


def update_parameters(parameters, grads, total_hidden_layer, learning_rate=1.2):
    new_parameters = parameters.copy()
    for i in range(1, total_hidden_layer + 1):
        W = new_parameters[f"W{i}"]
        b = new_parameters[f"b{i}"]

        dW = grads[f"dW{i}"]
        db = grads[f"db{i}"]

        W -= learning_rate * dW
        b -= learning_rate * db
        new_parameters[f"W{i}"] = W
        new_parameters[f"b{i}"] = b

    return new_parameters


def nn_model(X, y, layer_sizes, parameters, activations, number_of_iterations=10000, learning_rate=1.2, print_cost=False, lamd=0, keep_prob=1):
    assert len(layer_sizes) == len(activations) + 1

    cost_history = []
    for i in range(number_of_iterations):
        cache = forward_propagation(X, parameters, activations, keep_prob=keep_prob)
        # cost = compute_sigmoid_cost_with_regularization(cache[f"A{len(activations)}"], y, parameters, layer_sizes, lamd)
        cost = compute_sigmoid_cost(cache[f"A{len(activations)}"], y)
        grads = backward_propagation(X, y, parameters, cache, activations, lamd=lamd, keep_prob=keep_prob)
        parameters = update_parameters(
            parameters, grads, total_hidden_layer=len(activations), learning_rate=learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))

        if i % 100 == 0:
            cost_history.append(cost)
    
    parameters["learning_rate"] = learning_rate
    parameters["number_of_iterations"] = number_of_iterations
    parameters["activations"] = activations
    parameters["layer_sizes"] = layer_sizes
    return parameters, cost_history


def predict(parameters, activations, X, decision_rate=0.5):
    cache = forward_propagation(X, parameters, activations)
    return cache[f"A{len(activations)}"] > decision_rate

from sklearn.metrics import confusion_matrix

def metrics(y, predictions):
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
   
    precision = tp / (tp + fp) * 100
    recall = tp / (tp + fn) * 100
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100

    print(f"Precision: {precision} %")
    print(f"Recall: {recall} %")
    print(f"Accuracy: {accuracy} %")


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()
    # %%

#%%
