import sys
sys.path.append("../")
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
import matplotlib.colorizer as mcolorizer
import matplotlib.colors as mcolors
from models.neural_network_regressor import NeuralNetworkRegressor
from models.ridge import Ridge
from models.lasso import Lasso
from functions.activation_functions import ReLU, ReLU_der, LReLU, LReLU_der, \
                                            sigmoid, sigmoid_der
from functions.cost_functions import mse, mse_der
from functions.runge import runge
from types import NoneType

def plot_heatmap_ridge(n, r, epochs, gammas, lambdas, title1, title2, filename, rng):
    x_train = np.linspace(-1, 1, n).reshape(-1, 1)
    y_train = runge(x_train).reshape(-1, 1) + rng.normal(0, 0.1, size=n).reshape(-1, 1)

    x_test = np.linspace(-1, 1, int(r*n)).reshape(-1, 1)
    y_test = runge(x_test).reshape(-1, 1) + rng.normal(0, 0.1, size=int(r*n)).reshape(-1, 1)

    layer_output_sizes = [50, 50, 50, 1]
    activation_funcs = [sigmoid, sigmoid, sigmoid, sigmoid]
    activation_ders = [sigmoid_der, sigmoid_der, sigmoid_der, sigmoid_der]

    errs_net = np.zeros(shape=(len(lambdas), len(gammas)))
    errs_ridge = np.zeros(shape=(len(lambdas)))
    for i, lmbd in enumerate(lambdas):
        model = Ridge(lmbd, 8).fit(x_train, y_train)
        errs_ridge[i] = model.mse(x_test, y_test)
        for j, gamma in enumerate(gammas):
            net = NeuralNetworkRegressor(
                network_input_size=1,
                layer_output_sizes=layer_output_sizes,
                activation_funcs=activation_funcs,
                activation_ders=activation_ders,
                cost_fun=mse,
                cost_der=mse_der,
                lmbd2=lmbd,
                gamma=gamma,
                epochs=epochs,
                batch_size=10,
                method="adam",
                rng=rng
            )
            net.fit(x_train, y_train)
            errs_net[j, i] = net.mse(x_test, y_test)

    M = np.max((np.max(errs_net), np.max(errs_ridge)))
    m = np.min((np.min(errs_net), np.min(errs_ridge)))
    norm = mcolors.Normalize(vmin=m, vmax=M)
    colorizer = mcolorizer.Colorizer(norm=norm, cmap="viridis_r")

    fig = plt.figure(layout="compressed")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1/5], figure=fig)

    net_ax: Axes = plt.subplot(gs[0])
    im_net = net_ax.imshow(errs_net, colorizer=colorizer)
    net_ax.set_xticks(range(len(gammas)), labels=gammas)
    net_ax.set_yticks(range(len(lambdas)), labels=lambdas)
    errs_round = np.round(errs_net, 3)
    for i in range(len(lambdas)):
        for j in range(len(gammas)):
            text = net_ax.text(j, i, errs_round[i, j],
                        ha="center", va="center", color="k")
    net_ax.set_xlabel("Learning rate $\\gamma$")
    net_ax.set_ylabel("$L^2$ penalty $\\lambda$")
    net_ax.set_title(title1)

    ridge_ax: Axes = plt.subplot(gs[1])

    im_ridge = ridge_ax.imshow(errs_ridge.reshape(-1, 1), colorizer=colorizer)
    ridge_ax.set_xticks((), labels=(""))
    ridge_ax.set_yticks((), labels=(""))
    errs_round = np.round(errs_ridge, 3)
    for i in range(len(lambdas)):
        text = ridge_ax.text(0, i, errs_round[i],
                        ha="center", va="center", color="k")
    ridge_ax.set_title(title2)

    fig.colorbar(im_net, ax=(net_ax, ridge_ax), orientation="vertical")
    fig.savefig(filename)

def plot_heatmap_lasso(n, r, epochs, gammas, lambdas, title1, title2, filename, rng):
    x_train = np.linspace(-1, 1, n).reshape(-1, 1)
    y_train = runge(x_train).reshape(-1, 1) + rng.normal(0, 0.1, size=n).reshape(-1, 1)

    x_test = np.linspace(-1, 1, int(r*n)).reshape(-1, 1)
    y_test = runge(x_test).reshape(-1, 1) + rng.normal(0, 0.1, size=int(r*n)).reshape(-1, 1)

    layer_output_sizes = [50, 50, 50, 1]
    activation_funcs = [sigmoid, sigmoid, sigmoid, sigmoid]
    activation_ders = [sigmoid_der, sigmoid_der, sigmoid_der, sigmoid_der]

    errs_net = np.zeros(shape=(len(lambdas), len(gammas)))
    errs_lasso = np.zeros(shape=(len(lambdas), len(gammas)))
    for i, lmbd in enumerate(lambdas):
        for j, gamma in enumerate(gammas):
            model = Lasso(lmbd, 8, gamma=gamma, epochs=epochs, method="adam").fit(x_train, y_train)
            errs_lasso[j, i] = model.mse(x_test, y_test)

            net = NeuralNetworkRegressor(
                network_input_size=1,
                layer_output_sizes=layer_output_sizes,
                activation_funcs=activation_funcs,
                activation_ders=activation_ders,
                cost_fun=mse,
                cost_der=mse_der,
                lmbd1=lmbd,
                gamma=gamma,
                epochs=epochs,
                batch_size=10,
                method="adam",
                rng=rng
            )
            net.fit(x_train, y_train)
            errs_net[j, i] = net.mse(x_test, y_test)

    M = np.max((np.max(errs_net), np.max(errs_lasso)))
    m = np.min((np.min(errs_net), np.min(errs_lasso)))
    norm = mcolors.LogNorm(vmin=m, vmax=M)
    colorizer = mcolorizer.Colorizer(norm=norm, cmap="viridis_r")

    fig = plt.figure(layout="compressed", figsize=(6.4, 3.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)
    net_ax: Axes = plt.subplot(gs[0])
    im_net = net_ax.imshow(errs_net, colorizer=colorizer)
    net_ax.set_xticks(range(len(gammas)), labels=gammas)
    net_ax.set_yticks(range(len(lambdas)), labels=lambdas)
    errs_round = np.round(errs_net, 3)
    for i in range(len(lambdas)):
        for j in range(len(gammas)):
            text = net_ax.text(j, i, errs_round[i, j],
                        ha="center", va="center", color="k")
    net_ax.set_xlabel("Learning rate $\\gamma$")
    net_ax.set_ylabel("$L^1$ penalty $\\lambda$")
    net_ax.set_title(title1)

    lasso_ax: Axes = plt.subplot(gs[1])

    im_lasso = lasso_ax.imshow(errs_lasso, colorizer=colorizer)
    lasso_ax.set_xticks(range(len(gammas)), labels=gammas)
    lasso_ax.set_yticks((), labels=(""))
    errs_round = np.round(errs_lasso, 3)
    for i in range(len(lambdas)):
        for j in range(len(gammas)):
            text = lasso_ax.text(j, i, errs_round[i, j],
                        ha="center", va="center", color="k")
    lasso_ax.set_xlabel("Learning rate $\\gamma$")
    lasso_ax.set_title(title2)

    fig.colorbar(im_lasso, ax=(net_ax, lasso_ax), orientation="vertical", use_gridspec=True, fraction=0.05)
    fig.savefig(filename)

rng = np.random.default_rng(2026)
gammas = np.logspace(-4, 0, 5)
lambdas = np.logspace(-4, 0, 5)
n = 100
r = 0.25
# Only one hundred epochs to allow gamma to have effect on values
epochs=100

plot_heatmap_ridge(n, r, epochs, gammas, lambdas, "MSE of Neural Network", "MSE of Ridge", "..//results//nn_vs_ridge.png", rng)
plot_heatmap_lasso(n, r, epochs, gammas, lambdas, "MSE of Neural Network", "MSE of Lasso", "..//results//nn_vs_lasso.png", rng)