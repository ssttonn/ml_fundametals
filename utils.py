import numpy as np
import time

import pandas as pd


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

# def randomly_initialize_parameters(input_layer_sizes):
#     W1 = np.random.rand(4, 2) * 0.01
#     b1 = np.zeros((4, 1))
#     W2 = np.random.rand(1, 4) * 0.01
#     b2 = np.zeros((1, 1))
#     return {
#         "W1": W1,
#         "b1": b1,
#         "W2": W2,
#         "b2": b2
#     }


def forward_propagation(X, parameters, activations):
    cache = {}

    new_parameters = parameters.copy()

    # Forward propagation
    for i, activation in enumerate(activations):
        l = i + 1
        A_last = X if l == 1 else cache[f"A{l-1}"]
        W = new_parameters[f"W{l}"]
        b = new_parameters[f"b{l}"]
        Z = np.dot(W, A_last) + b
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = activation(Z)[0]

    return cache


# def forward_propagation(X, parameters, activations):
#     W1 = parameters["W1"]
#     b1 = parameters["b1"]
#     W2 = parameters["W2"]
#     b2 = parameters["b2"]
#
#     Z1 = np.dot(W1, X) + b1
#     A1 = tanh(Z1)[0]
#     Z2 = np.dot(W2, A1) + b2
#     A2 = sigmoid(Z2)[0]
#
#     cache = {
#         "Z1": Z1,
#         "A1": A1,
#         "Z2": Z2,
#         "A2": A2
#     }
#     return cache


def compute_sigmoid_cost(A_last, Y):
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y, np.log(A_last).T) - np.dot(1-Y, np.log(1-A_last).T))
    cost = np.squeeze(cost)
    return cost


# def backward_propagation(X, Y, parameters, cache, activations):
#     m = X.shape[1]
#     W2 = parameters["W2"]
#
#     A1 = cache["A1"]
#     A2 = cache["A2"]
#
#     dZ2 = A2 - Y
#     dW2 = (1 / m) * np.dot(dZ2, A1.T)
#     db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
#     dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
#     dW1 = (1 / m) * np.dot(dZ1, X.T)
#     db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
#
#     grads = {
#         "dW1": dW1,
#         "db1": db1,
#         "dW2": dW2,
#         "db2": db2
#     }
#     return grads

def backward_propagation(X, Y, parameters, cache, activations):
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
            dZ = np.dot(W_last.T, dZ_last) * activation(Z)[1]

        if current_layer_number == 1:
            A_next = X
        else:
            A_next = cache[f"A{current_layer_number - 1}"]

        dW = (1 / m) * np.dot(dZ, A_next.T)
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


def nn_model(X, y, layer_sizes, parameters, activations, number_of_iterations=10000, learning_rate=1.2, print_cost=False):
    assert len(layer_sizes) == len(activations) + 1

    cost_history = []
    for i in range(number_of_iterations):
        cache = forward_propagation(X, parameters, activations)
        cost = compute_sigmoid_cost(cache[f"A{len(activations)}"], y)
        grads = backward_propagation(X, y, parameters, cache, activations)
        parameters = update_parameters(
            parameters, grads, total_hidden_layer=len(activations), learning_rate=learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))

        if i % 100 == 0:
            cost_history.append(cost)

    return parameters, cost_history


def predict(parameters, activations, X, decision_rate=0.5):
    cache = forward_propagation(X, parameters, activations)
    return cache[f"A{len(activations)}"] > decision_rate


def metrics(y, predictions, n_h):
    print(f"Precision for n_h={n_h}: {float((np.dot(y, predictions.T))/(np.dot(y, predictions.T) + np.dot(1 - y, predictions.T)) * 100)} %")
    print(f"Recall for n_h={n_h}: {float((np.dot(y, predictions.T))/(np.dot(y, predictions.T) + np.dot(y, 1 - predictions.T)) * 100)} %")
    print(f"Accuracy for n_h={n_h}: {(float((np.dot(y, predictions.T) + np.dot(1 - y, 1 - predictions.T)) / float(y.size)) * 100)} %")
    # %%

#%%
