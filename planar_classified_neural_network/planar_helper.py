def layer_sizes(X, y):
    n_x = X.shape[0]
    n_h = 4
    n_y = y.shape[0]
    layers = (n_x, n_h, n_y)
    return layers