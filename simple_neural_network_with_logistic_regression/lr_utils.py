import copy
import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m

    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    return dw, db, cost


def gradient_descent(w, b, X, Y, num_iterations=100, learning_rate=0.09, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    cost_history = list()
    for i in range(num_iterations):
        dw, db, cost = propagate(w, b, X, Y)
        w -= learning_rate * dw
        b -= learning_rate * db
        if i % 100 == 0:
            cost_history.append(cost)

            # Print the cost every 100 training iterations
            if print_cost:
                print(f"Cost after iteration ${i}: ${cost}")

    return w, b, cost_history


def predict_w_b(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    return Y_prediction


def neural_network_logistic_model(X_train, Y_train, X_test, Y_test, learning_rate=0.5, print_cost=False,
                                  num_iterations=100):
    w, b = initialize_with_zeros(X_train.shape[0])
    w, b, cost_history = gradient_descent(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    Y_hat_train = predict_w_b(w, b, X_train)
    Y_hat_test = predict_w_b(w, b, X_test)
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_hat_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_hat_test - Y_test)) * 100))

    result = {"cost_history": cost_history,
              "Y_hat_test": Y_hat_test,
              "Y_hat_train": Y_hat_train,
              "w": w,
              "b": b,
              "learning_rate": learning_rate,
              "num_iterations": num_iterations}
    return result

# %%
