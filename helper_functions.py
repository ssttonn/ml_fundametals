import numpy as np
import matplotlib.pyplot as plt

def linear_regression(X, w, b):
    return np.dot(X, w.T) + b


def residual(X, w, b, y):
    return linear_regression(X, w, b) - y


def cost_function(X, w, b, y):
    m = X.shape[0]
    return np.sum(residual(X, w, b, y) ** 2) / (2 * m)


def compute_gradient(X, w, b, y):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    dj_dw += np.sum(residual(X, w, b, y) * X.T, axis=1)
    dj_db += np.sum(residual(X, w, b, y))
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


def gradient_descent(X, w, b, y, number_of_iterations, alpha):
    temp_w = w.copy()
    temp_b = b
    for _ in range(number_of_iterations):
        dj_dw, dj_db = compute_gradient(X, temp_w, temp_b, y)
        temp_w -= dj_dw * alpha
        temp_b -= dj_db * alpha
        # print(cost_function(X, temp_w, temp_b, y))

    return temp_w, temp_b


def plot_learning_curve(cost_history, learning_rate):
    plt.plot(cost_history)
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

#%%
