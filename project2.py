#our functions
import functions.activation_functions as activ_funcs
import functions.linear_regression as line_reg
import functions.score_analysis as scores

#python functions
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that results agree with numpy to ~1e-15
import jax.numpy as jnp                    # numpy-compatible functions that JAX can differentiate
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

activation_funcs = [activ_funcs.ReLU, activ_funcs.ReLU, activ_funcs.sigmoid]
input_size = 4
layer_output_sizes = [12, 10, 3]
inputs = np.random.rand(1000, input_size)

RUNGE_RUNS = (('plain', None, '#777777', 'plain GD'), ('momentum', None, '#004488', 'momentum, $\\beta=0.9$'), ('adagrad', 0.5, '#DDAA33', 'AdaGrad, $\\gamma=0.5$'), ('rmsprop', 0.01, '#228833', 'RMSProp, $\\gamma=0.01$'), ('adam', 0.05, '#BB5566', 'Adam, $\\gamma=0.05$'))

def feed_forward_batch(inputs, layers, activation_funcs):
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a @ W + b
        a = activation_func(z)
    return a

def cost(input, layers, activation_funcs, target):
    predict = feed_forward_batch(input, layers, activation_funcs)
    return cross_entropy(predict, target)

def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(i_size, layer_output_size)
        b = np.zeros(layer_output_size) + 0.01
        layers.append((W, b))

        i_size = layer_output_size
    return layers

def runge_data(n=100, degree=6, noise=0.1, seed=2026):
    """Runge function 1/(1+25x^2) on [-1,1], standardised polynomial features, centred y."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    y = 1.0 / (1.0 + 25.0 * x**2) + noise * rng.standard_normal(n)
    X = np.column_stack([x**k for k in range(1, degree + 1)])
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_norm, y - y.mean()

layers = create_layers_batch(input_size, layer_output_sizes)

print(layers)