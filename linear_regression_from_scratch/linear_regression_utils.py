import copy

import numpy as np
from matplotlib import pyplot as plt
from helper_functions import dldarkred, dlblue


def linear_function(X, w, b):
    return np.dot(X, w.T) + b


def cost_function(X, y, w, b):
    m = X.shape[0]
    return np.sum((linear_function(X, w, b) - y) ** 2) / (2 * m)


def compute_gradient(X, y, w, b):
    m = X.shape[0]
    loss = linear_function(X, w, b) - y
    dw = np.dot(loss, X)
    db = np.sum(loss)
    dw /= m
    db /= m
    return dw, db


def gradient_descent(X, y, w, b, learning_rate, print_cost=True, number_of_iterations=100):
    w_temp = copy.deepcopy(w)
    b_temp = copy.deepcopy(b)
    cost_history = []
    w_history = []
    b_history = []
    for i in range(number_of_iterations):
        dw, db = compute_gradient(X, y, w_temp, b_temp)
        w_temp -= learning_rate * dw
        b_temp -= learning_rate * db

        if print_cost:
            cost = cost_function(X, y, w_temp, b_temp)
            if i % 100 == 0 and i > 0:
                cost_history.append(cost)
                w_history.append(w_temp)
                b_history.append(b_temp)
                print(f"Cost for iterate no {i}: {cost} with w = {w_temp} and b = {b_temp}")
    return w_temp, b_temp, cost_history, w_history, b_history


def predict(X, w, b):
    return linear_function(X, w, b)


def linear_regression_model(X_train, y_train, X_test, y_test, initial_w=np.array([2.5]), initial_b=0.0, learning_rate=0.5, print_cost=False,
                            num_iterations=100):
    print("Initial weights:", initial_w)
    w, b, cost_history, w_history, b_history = gradient_descent(X_train, y_train, initial_w, initial_b, learning_rate, print_cost, num_iterations)

    result = {"cost_history": cost_history,
              "test_cost": cost_function(X_test, y_test, w, b),
              "train_cost": cost_function(X_train, y_train, w, b),
              "y_hat_test": predict(X_test, w, b),
              "y_hat_train": predict(X_train, w, b),
              "w": w,
              "b": b,
              "learning_rate": learning_rate,
              "num_iterations": num_iterations}
    return result


def add_line(dj_dx, x1, y1, d, ax):
    x = np.linspace(x1-d, x1+d,50)
    y = dj_dx*(x - x1) + y1
    ax.scatter(x1, y1, color=dlblue, s=50)
    ax.plot(x, y, '--', c=dldarkred,zorder=10, linewidth = 1)
    xoff = 30 if x1 == 200 else 10
    ax.annotate(r"$\frac{\partial J}{\partial w}$ =%d" % dj_dx, fontsize=14,
                xy=(x1, y1), xycoords='data',
                xytext=(xoff, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle="->"),
                horizontalalignment='left', verticalalignment='top')


def plt_gradients(X_train, y_train, f_compute_cost, w_start=-20, w_end=20, w_step=50, b_value=0.0, w_initial=np.array([10])):
    #===============
    #  First subplot
    #===============
    fig,ax = plt.subplots(1, 1, figsize=(14,8))

    # Print w vs cost to see minimum
    fix_b = b_value
    w_array = np.linspace(w_start, w_end, w_step)
    cost = np.zeros_like(w_array)

    for i in range(len(w_array)):
        tmp_w = w_array[i].reshape(X_train.shape[1])
        cost[i] = f_compute_cost(X_train, y_train, tmp_w, fix_b)
    ax.plot(w_array, cost, linewidth=1)
    ax.scatter(w_initial, f_compute_cost(X_train, y_train, w_initial, fix_b), color="r")
    ax.set_title("Cost vs w, with gradient; b set to 0")
    ax.set_ylabel('Cost')
    ax.set_xlabel('w')


# def plot_gradient_cost(X_train, y_train, compute_cost, cost_history, w_history, b_history):
#     fig,ax = plt.subplots(1, 2, figsize=(12,4))
#
#     # Print w vs cost to see minimum
#     fix_b = 0.0
#     w_array = np.linspace(-5, 6, 50)
#     cost = np.zeros_like(w_array)
#
#     for i in range(len(w_array)):
#         tmp_w = w_array[i].reshape(X_train.shape[1])
#         cost[i] = compute_cost(X_train, y_train, tmp_w, fix_b)
#     print(cost_history)
#     print(w_history)
#     print(b_history)
#     ax[0].scatter(np.array(cost_history), np.array(w_history), c='r', marker='x')
#     ax[0].plot(w_array, cost, linewidth=1)
#     ax[0].set_title("Cost vs w, with gradient")
#     ax[0].set_ylabel('Cost')
#     ax[0].set_xlabel('w')
#     ax[1].scatter(np.array(cost_history), np.array(cost_history), c='r', marker='x')
#     ax[1].plot(w_array, cost, linewidth=1)
#     ax[1].set_title("Cost vs b, with gradient")
#     ax[1].set_ylabel('Cost')
#     ax[1].set_xlabel('b')



#%%
