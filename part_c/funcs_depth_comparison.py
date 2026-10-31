import sys
sys.path.append("../")
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import matplotlib as mpl
from models.neural_network_regressor import NeuralNetworkRegressor
from models.ols import OLS
from functions.activation_functions import ReLU, ReLU_der, LReLU, LReLU_der, \
                                            sigmoid, sigmoid_der
from functions.cost_functions import mse, mse_der
from functions.runge import runge

def plot_heatmap(n, r, epochs, layer_depths, layer_sizes, activation_func, activation_der, title, filename, rng):
    x_train = np.linspace(-1, 1, n).reshape(-1, 1)
    y_train = runge(x_train).reshape(-1, 1) + rng.normal(0, 0.1, size=n).reshape(-1, 1)

    x_test = np.linspace(-1, 1, int(r*n)).reshape(-1, 1)
    y_test = runge(x_test).reshape(-1, 1) + rng.normal(0, 0.1, size=int(r*n)).reshape(-1, 1)

    # Neural network
    gamma = 0.01
    errs = np.zeros(shape=(len(layer_sizes), len(layer_depths)))
    for i, layer_size in enumerate(layer_sizes):
        for j, depth in enumerate(layer_depths):
            layer_output_sizes = [layer_size for _ in range(depth)] + [1]
            activation_funcs = [activation_func for _ in range(depth)]
            activation_ders = [activation_der for _ in range(depth)]
            net = NeuralNetworkRegressor(
                network_input_size=1,
                layer_output_sizes=layer_output_sizes,
                activation_funcs=activation_funcs,
                activation_ders=activation_ders,
                cost_fun=mse,
                cost_der=mse_der,
                gamma=gamma,
                epochs=epochs,
                batch_size=10,
                method="adam",
                rng=rng
            )
            net.fit(x_train, y_train)
            errs[i, j] = net.mse(x_test, y_test)

    fig, ax = plt.subplots()
    segmentdata = {
        'red':   [(0.0, 0.0, 0.0),
                (0.5, 1.0, 1.0),
                (1.0, 1.0, 1.0)],
        'green': [(0.0, 1.0, 0.75),
                (0.5, 1.0, 1.0),
                (1.0, 0.0, 0.0)],
        'blue':  [(0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (1.0, 0.0, 0.0)]
    }
    segmentdata = {
        'red':   [(0.0, 0.0, 0.0),
                (0.5, 1.0, 1.0),
                (1.0, 1.0, 0.0)],
        'green': [(0.0, 0.0, 0.75),
                (0.5, 1.0, 1.0),
                (1.0, 0.0, 0.0)],
        'blue':  [(0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (1.0, 0.0, 0.0)]
    }
    # I hope whoever is looking at these figures is not red-green colorblind!
    cmap = mpl.colors.LinearSegmentedColormap(name="GrYlRd", segmentdata=segmentdata)
    im = ax.imshow(errs, cmap=cmap)
    ax.set_xticks(range(len(layer_depths)), labels=layer_depths)
    ax.set_yticks(range(len(layer_sizes)), labels=layer_sizes)
    errs_round = np.round(errs, 3)
    for i in range(len(layer_sizes)):
        for j in range(len(layer_depths)):
            text = ax.text(j, i, errs_round[i, j],
                        ha="center", va="center", color="k")
    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Layer size")
    fig.colorbar(im, label="MSE")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename)

rng = np.random.default_rng(2026)
gammas = np.logspace(-4, 0, 9)
n = 100
r = 0.25
# Only 100 epochs, so script can finish before heat death of the universe
epochs=100
layer_sizes = [10, 25, 50, 100, 500]

plot_heatmap(n, r, epochs, [1, 2, 3, 5, 10], layer_sizes, sigmoid, sigmoid_der, "MSE of Neural Network with Sigmoid Activation Functions", "..//results//sigmoid_heatmap.png", rng)
rng = np.random.default_rng(2026)
plot_heatmap(n, r, epochs, [1, 2, 3, 5, 10], layer_sizes, ReLU, ReLU_der, "MSE of Neural Network with ReLU Activation Functions", "..//results//relu_heatmap.png", rng)
rng = np.random.default_rng(2026)
plot_heatmap(n, r, epochs, [1, 2, 3, 5, 10], layer_sizes, LReLU, LReLU_der, "MSE of Neural Network with Leaky ReLU Activation Functions", "..//results//lrelu_heatmap.png", rng)