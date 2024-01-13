import copy

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calculate_gradient(X, y, w, b):
    m = X.shape[0]

    y_hat = sigmoid(np.dot(X, w.T) + b)
    cost = -1 * np.sum((y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))) / m

    dw = np.dot(y_hat - y, X) / m
    db = np.sum(y_hat - y) / m

    return dw, db, cost


def gradient_descent(X, y, w, b, learning_rate=0.005, number_of_iterations=100, log_cost=False):
    w_temp = copy.deepcopy(w)
    b_temp = copy.deepcopy(b)
    cost_history = list()
    for i in range(number_of_iterations):
        dw, db, cost = calculate_gradient(X, y, w_temp, b_temp)
        w_temp -= learning_rate * dw
        b_temp -= learning_rate * db

        if i % 100 == 0:
            cost_history.append(cost)

            # Print the cost every 100 training iterations
            if log_cost:
                print(f"Cost after iteration ${i}: ${cost}")
    return w_temp, b_temp, cost_history


def predict(X, w, b):
    m = X.shape[0]
    y_prediction = np.zeros(m)

    y_hat = sigmoid(np.dot(X, w.T) + b)
    print(f"Pridiction: {y_prediction}")
    for i in range(y_hat.shape[0]):
        if y_hat[i] > 0.5:
            y_prediction[i] = 1
        else:
            y_prediction[i] = 0
    return y_prediction


def logistic_regression_model(X_train, y_train, X_test, y_test, learning_rate=0.5, print_cost=False,
                              num_iterations=100):
    w = np.zeros(X_train.shape[1])
    b = 0
    w_result, b_result, cost_history = gradient_descent(X_train, y_train, w, b, number_of_iterations=num_iterations,
                                          learning_rate=learning_rate, log_cost=print_cost)
    y_hat_train = predict(X_train, w_result, b_result)
    y_hat_test = predict(X_test, w_result, b_result)
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(y_hat_train - y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(y_hat_test - y_test)) * 100))

    result = {"cost_history": cost_history,
              "Y_hat_test": y_hat_test,
              "Y_hat_train": y_hat_train,
              "w": w,
              "b": b,
              "learning_rate": learning_rate,
              "num_iterations": num_iterations}
    return result
# %%
