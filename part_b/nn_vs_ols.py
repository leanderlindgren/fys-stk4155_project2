import sys
sys.path.append("../")
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
from models.neural_network_regressor import NeuralNetworkRegressor
from models.ols import OLS
from functions.activation_functions import sigmoid, sigmoid_der
from functions.cost_functions import mse, mse_der
from functions.runge import runge

def plot_nn_vs_ols(n, r, gammas, epochs, layer_output_sizes, activation_funcs, activation_ders, title, filename, rng):
    # OLS
    errs = []
    models = []

    x_train = np.linspace(-1, 1, n).reshape(-1, 1)
    y_train = runge(x_train).reshape(-1, 1) + rng.normal(0, 0.1, size=n).reshape(-1, 1)

    x_test = np.linspace(-1, 1, int(r*n)).reshape(-1, 1)
    y_test = runge(x_test).reshape(-1, 1) + rng.normal(0, 0.1, size=int(r*n)).reshape(-1, 1)

    for degree in range(1, 16):
        model = OLS(degree)
        model.fit(x_train, y_train)
        MSE = model.mse(x_test, y_test)

        errs.append(MSE)
        models.append(model)

    best_idx = np.array(errs).argmin()
    best_err = errs[best_idx]
    best_model = models[best_idx]

    fig, ax = plt.subplots()
    ax.plot([gammas[0], gammas[-1]], [best_err, best_err], color="r", linestyle="dashed", label=f"OLS, degree {best_model.degree}")

    # Neural network
    methods = ["Plain", "RMSProp", "Adam"]
    for method in methods:
        errs_stoch = []
        errs_nonstoch = []
        for gamma in gammas:
            # Stochastic
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
                method=method.lower(),
                rng=rng
            )
            net.fit(x_train, y_train)
            errs_stoch.append(net.mse(x_test, y_test))

            # Non-stochastic
            net = NeuralNetworkRegressor(
                network_input_size=1,
                layer_output_sizes=layer_output_sizes,
                activation_funcs=activation_funcs,
                activation_ders=activation_ders,
                cost_fun=mse,
                cost_der=mse_der,
                gamma=gamma,
                epochs=epochs,
                batch_size=n,
                method=method.lower(),
                rng=rng
            )
            net.fit(x_train, y_train)
            errs_nonstoch.append(net.mse(x_test, y_test))

        ax.plot(gammas, errs_stoch, label=f"{method}, stochastic")
        ax.plot(gammas, errs_nonstoch, label=f"{method}, non-stochastic")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Mean square error")
    fig.tight_layout()
    fig.savefig(filename)

rng = np.random.default_rng(2026)
gammas = np.logspace(-4, 0, 9)
n = 100
r = 0.25
epochs=1000

plot_nn_vs_ols(n, r, gammas, epochs, [50, 1], [sigmoid, sigmoid], [sigmoid_der, sigmoid_der], "Neural Network vs OLS, one hidden layer with 50 nodes", "..//results//nn_vs_ols_one_hidden_50", rng)
# Reset rng so that training data remains consistent
rng = np.random.default_rng(2026)
plot_nn_vs_ols(n, r, gammas, epochs, [50, 50, 1], [sigmoid, sigmoid], [sigmoid_der, sigmoid_der], "Neural Network vs OLS, two hidden layers with 50 nodes", "..//results//nn_vs_ols_two_hidden_50", rng)
rng = np.random.default_rng(2026)
plot_nn_vs_ols(n, r, gammas, epochs, [100, 1], [sigmoid, sigmoid], [sigmoid_der, sigmoid_der], "Neural Network vs OLS, one hidden layer with 100 nodes", "..//results//nn_vs_ols_one_hidden_100", rng)
rng = np.random.default_rng(2026)
plot_nn_vs_ols(n, r, gammas, epochs, [100, 100, 1], [sigmoid, sigmoid], [sigmoid_der, sigmoid_der], "Neural Network vs OLS, two hidden layers with 100 nodes", "..//results//nn_vs_ols_two_hidden_100", rng)