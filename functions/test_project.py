import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from cost_functions import mse, mse_jax, mse_der, mse_Lp, cross_entropy, \
        cross_entropy_Lp, cross_entropy_multiclass
from activation_functions import ReLU, ReLU_der, LReLU, LReLU_der, sigmoid, \
        sigmoid_der, softmax, ReLU_jax, LReLU_jax, sigmoid_jax, softmax_jax
import sys
sys.path.append("../")
from neural_network.neural_network import NeuralNetwork
from neural_network.neural_network_regressor import NeuralNetworkRegressor
from functions.runge import runge

def test_mse():
    """
    Test that the mse and mse_jax functions returns correct MSE
    """
    y = np.array([0.8, 0.2, -1.9, 1.4, 0.6, -0.3, -0.3, 0.3, -0.3, -0.2])
    y_pred = np.array([0.7, 0.5, -0.1, -0.1, 0.2, -0.6, -0.4, 0.5, -0.1, -1.4])
    expected = 0.737
    computed = mse(y_pred, y)

    msg = f"Computed MSE {computed} does not equal expected MSE {expected}"
    assert abs(expected - computed) < 1e-12, msg
    computed = mse_jax(y_pred, y)

    msg = f"JAX computed MSE {computed} does not equal expected MSE {expected}"
    assert abs(expected - computed) < 1e-12, msg

def test_mse_der():
    """
    Test that the mse_der function returns correct gradient of MSE
    """
    y = np.array([0.8, 0.2, -1.9, 1.4, 0.6, -0.3, -0.3, 0.3, -0.3, -0.2])
    y_pred = np.array([0.7, 0.5, -0.1, -0.1, 0.2, -0.6, -0.4, 0.5, -0.1, -1.4])
    expected = (1/5)*(y_pred - y)
    computed = mse_der(y_pred, y)

    msg = f"Computed gradient of MSE {computed} does not equal expected gradient of MSE {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

def test_mse_Lp():
    """
    Test that the mse_Lp function gives correct L1 and L2 penalty
    """
    y = np.array([0.8, 0.2, -1.9, 1.4, 0.6, -0.3, -0.3, 0.3, -0.3, -0.2])
    y_pred = np.array([0.7, 0.5, -0.1, -0.1, 0.2, -0.6, -0.4, 0.5, -0.1, -1.4])

    W1 = np.array([
        [-1, 2, -3],
        [4, 5, -6],
        [-7, 8, 9]
    ])
    b1 = np.array([
        [10, -11, 12]
    ])
    W2 = np.array([
        [13, -14, -15]
    ])
    b2 = np.array([
        [16]
    ])
    layers = [(W1, b1), (W2, b2)]

    expected = (np.sqrt(np.sum(abs(np.arange(1, 17)))))
    computed = mse_Lp(y_pred, y, layers, 1, 1) - mse(y_pred, y)

    msg = f"Computed L1 norm {computed} does not equal expected L1 norm {expected}"
    assert abs(expected - computed) < 1e-12, msg

    expected = (np.sqrt(np.sum(np.arange(1, 17)**2)))
    computed = mse_Lp(y_pred, y, layers, 1, 2) - mse(y_pred, y)

    msg = f"Computed L2 norm {computed} does not equal expected L2 norm {expected}"
    assert abs(expected - computed) < 1e-12, msg

def test_cross_entropy():
    """
    Test that the cross_entropy function gives correct cross entropy
    """
    y = np.array([0.2, 0.6, 0.5, 0.4, 0.4, 0.8, 0.9, 0.2, 0.7, 0.3])
    y_pred = np.array([1, 0.9, 0.6, 0.8, 0.5, 0.8, 0.4, 0.3, 0.3, 0.2])

    expected = np.sum(-y*np.log(y_pred + 1e-12))
    computed = cross_entropy(y_pred, y)

    msg = f"Computed cross entropy {computed} does not equal expected cross entropy {expected}"
    assert abs(expected - computed) < 1e-12, msg

def test_cross_entropy_Lp():
    """
    Test that the cross_entropy_Lp function gives correct L1 and L2 penalty
    """
    y = np.array([0.2, 0.6, 0.5, 0.4, 0.4, 0.8, 0.9, 0.2, 0.7, 0.3])
    y_pred = np.array([1, 0.9, 0.6, 0.8, 0.5, 0.8, 0.4, 0.3, 0.3, 0.2])

    W1 = np.array([
        [-1, 2, -3],
        [4, 5, -6],
        [-7, 8, 9]
    ])
    b1 = np.array([
        [10, -11, 12]
    ])
    W2 = np.array([
        [13, -14, -15]
    ])
    b2 = np.array([
        [16]
    ])
    layers = [(W1, b1), (W2, b2)]

    expected = (np.sqrt(np.sum(abs(np.arange(1, 17)))))
    computed = cross_entropy_Lp(y_pred, y, layers, 1, 1) - cross_entropy(y_pred, y)

    msg = f"Computed L1 norm {computed} does not equal expected L1 norm {expected}"
    assert abs(expected - computed) < 1e-12, msg

    expected = (np.sqrt(np.sum(np.arange(1, 17)**2)))
    computed = cross_entropy_Lp(y_pred, y, layers, 1, 2) - cross_entropy(y_pred, y)

    msg = f"Computed L2 norm {computed} does not equal expected L2 norm {expected}"
    assert abs(expected - computed) < 1e-12, msg

def test_cross_entropy_multiclass():
    """
    Test that the cross_entropy_multiclass function gives correct cross entropy
    """
    y = np.array([
        [0.2, 0.6, 0.5, 0.4, 0.4, 0.8, 0.9, 0.2, 0.7, 0.3],
        [0.5, 0.4, 0.7, 0.01, 0.4, 0.4, 0.2, 0.6, 0.4, 0.3]
    ]).T
    y_pred = np.array([
        [1, 0.9, 0.6, 0.8, 0.5, 0.8, 0.4, 0.3, 0.3, 0.2],
        [0.2, 0.9, 0.8, 0.6, 0.3, 0.9, 0.6, 0.4, 0.9, 0.3]
    ]).T

    expected = np.mean(np.sum(-y*np.log(y_pred + 1e-12), axis=1))
    computed = cross_entropy_multiclass(y_pred, y)

    msg = f"Computed cross entropy {computed} does not equal expected cross entropy {expected}"
    assert abs(expected - computed) < 1e-12, msg

def test_ReLU():
    """
    Test that ReLU and ReLU_jax functions calculate correct values
    """
    z = np.array([0.8, 0.2, -1.9, 1.4, 0.6, -0.3, -0.3, 0.3, -0.3, -0.2])
    expected = np.array([0.8, 0.2, 0.0, 1.4, 0.6, 0.0, 0.0, 0.3, 0.0, 0.0])
    computed = ReLU(z)

    msg = f"Computed ReLU(z) {computed} does not equal expected ReLU(z) {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

    computed = ReLU_jax(z)

    msg = f"Computed ReLU_jax(z) {computed} does not equal expected ReLU_jax(z) {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

def test_ReLU_der():
    """
    Test that ReLU_der function calculates correct values
    """
    z = np.array([0.8, 0.2, -1.9, 1.4, 0.6, -0.3, -0.3, 0.3, -0.3, -0.2])
    expected = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    computed = ReLU_der(z)

    msg = f"Computed ReLU_der(z) {computed} does not equal expected ReLU_der(z) {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

def test_LReLU():
    """
    Test that ReLU and ReLU_jax functions calculate correct values
    """
    z = np.array([0.8, 0.2, -1.9, 1.4, 0.6, -0.3, -0.3, 0.3, -0.3, -0.2])
    expected = np.array([0.8, 0.2, -0.019, 1.4, 0.6, -0.003, -0.003, 0.3, -0.003, -0.002])
    computed = LReLU(z)

    msg = f"Computed LReLU(z) {computed} does not equal expected LReLU(z) {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

    computed = LReLU_jax(z, alpha=0.01)

    msg = f"Computed LReLU_jax(z) {computed} does not equal expected LReLU_jax(z) {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

def test_ReLU_der():
    """
    Test that ReLU_der function calculates correct values
    """
    z = np.array([0.8, 0.2, -1.9, 1.4, 0.6, -0.3, -0.3, 0.3, -0.3, -0.2])
    expected = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    computed = ReLU_der(z)

    msg = f"Computed ReLU_der(z) {computed} does not equal expected ReLU_der(z) {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

def test_LReLU_der():
    """
    Test that LReLU_der function calculates correct values
    """
    z = np.array([0.8, 0.2, -1.9, 1.4, 0.6, -0.3, -0.3, 0.3, -0.3, -0.2])
    expected = np.array([1.0, 1.0, -0.01, 1.0, 1.0, -0.01, -0.01, 1.0, -0.01, -0.01])
    computed = LReLU_der(z)

    msg = f"Computed LReLU_der(z) {computed} does not equal expected LReLU_der(z) {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

def test_sigmoid():
    """
    Test that sigmoid and sigmoid_jax functions calculate correct values
    """
    z = np.array([-999, 0, 999, np.log(2.2), np.log(1.2), np.log(0.2), np.log(4.5), np.log(1.8), np.log(0.7), np.log(0.8)])
    expected = np.array([0, 0.5, 1, 0.6875, 6/11, 1/6, 9/11, 9/14, 7/17, 4/9])
    computed = sigmoid(z)

    msg = f"Computed sigmoid(z) {computed} does not equal expected sigmoid(z) {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

    computed = sigmoid_jax(z)

    msg = f"Computed sigmoid_jax(z) {computed} does not equal expected sigmoid_jax(z) {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

def test_sigmoid_der():
    """
    Test that sigmoid_der function calculates correct values
    """
    z = np.array([-999, 0, 999, np.log(2.2), np.log(1.2), np.log(0.2), np.log(4.5), np.log(1.8), np.log(0.7), np.log(0.8)])
    expected = np.array([0, 0.25, 0, 55/256, 30/121, 5/36, 18/121, 45/196, 70/289, 20/81])
    computed = sigmoid_der(z)

    msg = f"Computed sigmoid(z) {computed} does not equal expected sigmoid(z) {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

def test_softmax():
    """
    Test that softmax and softmax_jax functions calculate correct values
    """
    z = np.array([
        [np.log(1.2), np.log(1.8), np.log(1.6), np.log(1.5), np.log(1.5), np.log(2.2), np.log(2.5), np.log(1.2), np.log(2.0), np.log(1.3)],
        [np.log(1.7), np.log(1.5), np.log(2.0), np.log(1.0), np.log(1.5), np.log(1.5), np.log(1.2), np.log(1.8), np.log(1.5), np.log(1.3)]
    ]).T

    expected = np.array([
        [1.2/(1.2+1.7), 1.7/(1.2+1.7)],
        [1.8/(1.8+1.5), 1.5/(1.8+1.5)],
        [1.6/(1.6+2.0), 2.0/(1.6+2.0)],
        [1.5/(1.5+1.0), 1.0/(1.5+1.0)],
        [1.5/(1.5+1.5), 1.5/(1.5+1.5)],
        [2.2/(2.2+1.5), 1.5/(2.2+1.5)],
        [2.5/(2.5+1.2), 1.2/(2.5+1.2)],
        [1.2/(1.2+1.8), 1.8/(1.2+1.8)],
        [2.0/(2.0+1.5), 1.5/(2.0+1.5)],
        [1.3/(1.3+1.3), 1.3/(1.3+1.3)]
    ])
    computed = softmax(z)

    msg = f"Computed softmax(z) {computed} does not equal expected softmax(z) {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

    computed = softmax_jax(z)

    msg = f"Computed softmax_jax(z) {computed} does not equal expected softmax_jax(z) {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

def test_NN_create_layers():
    """
    Test that the create_layers method of the NeuralNetwork class
    generated layers with the correct shapes, and initializes correct biases
    """
    rng = np.random.default_rng(2026)
    n_layers = 5
    network_input_size = rng.integers(1, 20, 1)[0]
    layer_output_sizes = rng.integers(1, 20, n_layers)
    activation_funcs = [ReLU for _ in range(n_layers)]
    nn = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, None, None, None)
    nn.layers_ = nn._create_layers(network_input_size, layer_output_sizes)
    expected = [((network_input_size, layer_output_sizes[0]), layer_output_sizes[0])]
    expected += [((layer_output_sizes[i], layer_output_sizes[i+1]), layer_output_sizes[i+1]) for i in range(n_layers-1)]
    computed = []
    for (W, b) in nn.layers_:
        computed.append((W.shape, len(b)))

    msg = f"Computed layer shapes {computed} do not equal expected layer shapes {expected}"
    assert computed == expected, msg

    computed = [np.all(abs(b - 0.01) < 1e-12) for _, b in nn.layers_]
    msg = f"Computed biases do not equal expected biases"
    assert not False in computed, msg

def test_NN_predict():
    """
    Test that the predict and jax_compliant_predict methods
    of the NeuralNetwork class provides correct predictions
    """
    rng = np.random.default_rng(2026)
    n_layers = 2
    network_input_size = rng.integers(1, 20, 1)[0]
    layer_output_sizes = rng.integers(1, 20, n_layers)
    activation_funcs = [ReLU, sigmoid]
    scale = np.sqrt(2.0 / network_input_size)
    W_1 = rng.normal(scale=scale, size=(network_input_size, layer_output_sizes[0]))
    b_1 = np.zeros(layer_output_sizes[0]) + 0.01
    scale = np.sqrt(1.0 / layer_output_sizes[0])
    W_2 = rng.normal(scale=scale, size=(layer_output_sizes[0], layer_output_sizes[1]))
    b_2 = np.zeros(layer_output_sizes[1]) + 0.01

    layers = [(W_1, b_1), (W_2, b_2)]

    x = rng.normal(size=(100, network_input_size))
    a1 = activation_funcs[0](x@W_1 + b_1)
    expected = activation_funcs[1](a1@W_2 + b_2)

    nn = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, None, None, None)
    nn.layers_ = layers
    computed = nn.predict(x)

    msg = f"Computed predictions {computed} do not equal expected predictions {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

    computed = nn.jax_compliant_predict(nn.layers_, x)

    msg = f"JAX computed predictions {computed} do not equal expected predictions {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

def test_NN_cost():
    """
    Test that the cost and jax_compliant_cost method of the NeuralNetwork class
    correctly calculate cost with no penalty, and with L1 and L2 penalty
    """
    rng = np.random.default_rng(2026)
    n_layers = 5
    network_input_size = rng.integers(1, 20, 1)[0]
    layer_output_sizes = rng.integers(1, 20, n_layers)
    activation_funcs = [ReLU, sigmoid]

    inputs = rng.normal(size=(100, network_input_size))
    targets = runge(inputs)
    nn = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, None, mse, None)
    nn.layers_ = nn._create_layers(network_input_size, layer_output_sizes)
    computed = nn.cost(inputs, targets)
    predicts = nn.predict(inputs)
    expected = mse(predicts, targets)

    msg = f"Computed cost {computed} does not equal expected cost {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

    computed = nn.jax_compliant_cost(nn.layers_, inputs, targets, 0.0, 0.0)

    msg = f"JAX computed cost {computed} does not equal expected cost {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

    L1_penalty = sum([np.sum(abs(W)) for (W, _) in nn.layers_])
    computed = nn.cost(inputs, targets, lam1=1.0)
    expected = nn.cost(inputs, targets) + L1_penalty

    msg = f"Computed L1 cost {computed} does not equal expected L1 cost {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

    computed = nn.jax_compliant_cost(nn.layers_, inputs, targets, 1.0, 0.0)

    msg = f"JAX computed L1 cost {computed} does not equal expected L1 cost {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

    L2_penalty = sum([np.sum(W**2) for (W, _) in nn.layers_])
    computed = nn.cost(inputs, targets, lam2=1.0)
    expected = nn.cost(inputs, targets) + L2_penalty

    msg = f"Computed L2 cost {computed} does not equal expected L2 cost {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

    computed = nn.jax_compliant_cost(nn.layers_, inputs, targets, 0.0, 1.0)

    msg = f"JAX computed L2 cost {computed} does not equal expected L2 cost {expected}"
    assert np.sum(abs(expected - computed)) < 1e-12, msg

def test_NN__feed_forward_saver():
    """
    Test that the _feed_forward_saver method of the NeuralNetwork class
    provides correct unactivated layer outputs, layer inputs and predictions
    """
    rng = np.random.default_rng(2026)
    n_layers = 2
    network_input_size = rng.integers(1, 20, 1)[0]
    layer_output_sizes = rng.integers(1, 20, n_layers)
    activation_funcs = [ReLU, sigmoid]
    scale = np.sqrt(2.0 / network_input_size)
    W_1 = rng.normal(scale=scale, size=(network_input_size, layer_output_sizes[0]))
    b_1 = np.zeros(layer_output_sizes[0]) + 0.01
    scale = np.sqrt(1.0 / layer_output_sizes[0])
    W_2 = rng.normal(scale=scale, size=(layer_output_sizes[0], layer_output_sizes[1]))
    b_2 = np.zeros(layer_output_sizes[1]) + 0.01

    layers = [(W_1, b_1), (W_2, b_2)]

    x = rng.normal(size=(100, network_input_size))
    z1 = x@W_1 + b_1
    a1 = activation_funcs[0](z1)
    z2 = a1@W_2 + b_2
    expected = activation_funcs[1](z2)

    nn = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, None, None, None)
    nn.layers_ = layers
    layer_inputs, zs, predicts = nn._feed_forward_saver(x)

    computed = layer_inputs
    expected = [x, a1]
    s = sum([np.sum(abs(c - e)) for c, e in zip(computed, expected)])

    msg = f"Computed layer inputs {computed} do not equal expected layer inputs {expected}"
    assert s < 1e-12, msg

    computed = zs
    expected = [z1, z2]
    s = sum([np.sum(abs(c - e)) for c, e in zip(computed, expected)])

    msg = f"Computed unactivated outputs {computed} do not equal expected unactivated outputs {expected}"
    assert s < 1e-12, msg

    computed = predicts
    expected = nn.predict(x)

    msg = f"Computed predicts {computed} do not equal expected predicts {expected}"
    assert np.sum(abs(computed - expected)) < 1e-12, msg

def test_NN_compute_gradient():
    """
    Test that the compute_gradients method of the
    NeuralNetwork class provides correct gradients
    """
    rng = np.random.default_rng(2026)
    n_layers = 5
    network_input_size = rng.integers(1, 20, 1)[0]
    layer_output_sizes = rng.integers(1, 20, n_layers)
    layer_output_sizes[-1] = network_input_size
    activation_funcs = [ReLU_jax, ReLU_jax, ReLU_jax, ReLU_jax, sigmoid_jax]
    activation_ders = [ReLU_der, ReLU_der, ReLU_der, ReLU_der, sigmoid_der]

    x = rng.normal(size=(100, network_input_size))
    y = runge(x)

    nn = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, activation_ders, mse_jax, mse_der)
    nn.layers_ = nn._create_layers(network_input_size, layer_output_sizes)
    computed = nn.compute_gradient(x, y)
    cost_grad = jax.grad(nn.jax_compliant_cost, argnums=0)
    expected = cost_grad(
        nn.layers_,
        x,
        y,
        0.0,
        0.0
    )

    s = sum([np.sum(abs(Wc - We)) + np.sum(abs(bc - be)) for (Wc, bc), (We, be) in zip(computed, expected)])

    msg = f"Computed gradients {computed} do not equal expected gradients {expected}"
    assert s < 1e-12, msg

    computed = nn.compute_gradient(x, y, lam1=1.0)
    cost_grad = jax.grad(nn.jax_compliant_cost, argnums=0)
    expected = cost_grad(
        nn.layers_,
        x,
        y,
        1.0,
        0.0
    )

    s = sum([np.sum(abs(Wc - We)) + np.sum(abs(bc - be)) for (Wc, bc), (We, be) in zip(computed, expected)])

    msg = f"Computed gradients of L1 cost {computed} do not equal expected gradients of L1 cost {expected}"
    assert s < 1e-12, msg

    computed = nn.compute_gradient(x, y, lam2=1.0)
    cost_grad = jax.grad(nn.jax_compliant_cost, argnums=0)
    expected_old = expected
    expected = cost_grad(
        nn.layers_,
        x,
        y,
        0.0,
        1.0
    )

    s = sum([np.sum(abs(Wc - We)) + np.sum(abs(bc - be)) for (Wc, bc), (We, be) in zip(computed, expected)])

    msg = f"Computed gradients of L2 cost {computed} do not equal expected gradients of L2 cost {expected}"
    assert s < 1e-12, msg

def test_update_weights():
    """
    Test that the update_weights method of the
    NeuralNetwork class correctly updates weights
    """
    # This test is really just an implementation of update_weights,
    # but it exists so that if any future changes to
    # update_weights breaks it, it will be caught.
    rng = np.random.default_rng(2026)
    n_layers = 5
    gammas = [0.01, 0.1, 1]
    network_input_size = rng.integers(1, 20, 1)[0]
    layer_output_sizes = rng.integers(1, 20, n_layers)
    layer_output_sizes[-1] = network_input_size
    activation_funcs = [ReLU_jax, ReLU_jax, ReLU_jax, ReLU_jax, sigmoid_jax]
    activation_ders = [ReLU_der, ReLU_der, ReLU_der, ReLU_der, sigmoid_der]

    x = rng.normal(size=(100, network_input_size))
    y = runge(x)

    nn = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, activation_ders, mse_jax, mse_der)
    nn.layers_ = nn._create_layers(network_input_size, layer_output_sizes)
    layer_grads = nn.compute_gradient(x, y)

    for gamma in gammas:
        expected = [
            (W - gamma*dW, b - gamma*db)
            for (W, b), (dW, db) in zip(nn.layers_, layer_grads)
        ]
        nn.update_weights(layer_grads, gamma)
        computed = nn.layers_

        s = sum([np.sum(abs(Wc - We)) + np.sum(abs(bc - be)) for (Wc, bc), (We, be) in zip(computed, expected)])

        msg = f"Computed gradients of L2 cost {computed} do not equal expected gradients of L2 cost {expected}"
        assert s < 1e-12, msg

def test_NNR_split_minibatches():
    """
    Test that the split_minibatches method of the
    NeuralNetworkRegressor class splits correctly
    """
    rng = np.random.default_rng(2026)
    n = 10000
    d = 10
    outputs = 3
    minibatch_size = 700
    x = rng.normal(size=(n, d))
    y = rng.normal(size=(n, outputs))

    net = NeuralNetworkRegressor(d, [outputs], [ReLU], [ReLU_der], mse, mse_der, batch_size=minibatch_size)
    minibatches = net.split_minibatches(x, y)

    expected = n + minibatch_size - n%minibatch_size
    computed = 0
    for (inputs, targets) in minibatches:
        msg = f"Inputs (shape {inputs.shape}) has different size than targets (shape {targets.shape})"
        assert inputs.shape[0] == targets.shape[0], msg

        msg = f"Minibatch inputs (shape {inputs.shape} has different number of features than expected {d})"
        assert inputs.shape[1] == d, msg

        msg = f"Minibatch targets (shape {targets.shape} has different number of outputs than expected {outputs})"
        assert targets.shape[1] == outputs, msg

        computed += inputs.shape[0]

    msg = f"Computed sum of minibatch sizes {computed} does not equal expected {expected}"
    assert expected == computed, msg

def test_NNR_optimiser_step():
    """
    Test that the optimiser_step method of the
    NeuralNetworkRegressor computes correct adjusted gradients
    """
    rng = np.random.default_rng(2026)
    n = 10000
    d = 10
    gamma = 0.03
    lam1 = 0.0
    lam2 = 0.0
    eps = 1e-8
    outputs = 3
    x = rng.normal(size=(n, d))
    y = rng.normal(size=(n, outputs))

    net = NeuralNetworkRegressor(d, [outputs], [ReLU], [ReLU_der], mse, mse_der, gamma=0.03)
    net.nn_.initialize_layers()
    layer_grads = net.nn_.compute_gradient(x, y, lam1, lam2)

    # Plain gradient
    state = {}
    computed = net.optimiser_step(
        layer_grads,
        state,
        3,
        method="plain",
    )
    expected = [(gamma*dW, gamma*db) for (dW, db) in layer_grads]

    s = sum([np.sum(abs(Wc - We)) + np.sum(abs(bc - be)) for (Wc, bc), (We, be) in zip(computed, expected)])

    msg = f"Computed adjusted gradients {computed} do not equal expected adjusted gradients {expected}"
    assert s < 1e-12, msg

    # RMSProp gradient
    rho = 0.99
    rW = abs(rng.normal(size=(d, outputs)))
    rb = abs(rng.normal(size=(outputs)))
    state = {"rW_1": rW, "rb_1": rb}
    rW = rho*rW + (1.0 - rho)*layer_grads[0][0]**2
    rb = rho*rb + (1.0 - rho)*layer_grads[0][1]**2
    computed = net.optimiser_step(
        layer_grads,
        state,
        3,
        method="rmsprop",
    )
    expected = [(gamma*dW/(np.sqrt(rW) + eps), gamma*db/(np.sqrt(rb) + eps)) for (dW, db) in layer_grads]

    s = sum([np.sum(abs(Wc - We)) + np.sum(abs(bc - be)) for (Wc, bc), (We, be) in zip(computed, expected)])

    msg = f"Computed adjusted gradients {computed} do not equal expected adjusted gradients {expected}"
    assert s < 1e-12, msg

    # Adam gradient
    beta1 = 0.9
    beta2 = 0.999
    t = 3
    mW = abs(rng.normal(size=(d, outputs)))
    mb = abs(rng.normal(size=(outputs)))
    rW = abs(rng.normal(size=(d, outputs)))
    rb = abs(rng.normal(size=(outputs)))
    state = {"rW_1": rW, "rb_1": rb, "mW_1": mW, "mb_1": mb}
    mW = beta1*mW + (1.0 - beta1)*layer_grads[0][0]
    mb = beta1*mb + (1.0 - beta1)*layer_grads[0][1]
    rW = beta2*rW + (1.0 - beta2)*layer_grads[0][0]**2
    rb = beta2*rb + (1.0 - beta2)*layer_grads[0][1]**2
    computed = net.optimiser_step(
        layer_grads,
        state,
        3,
        method="adam",
    )
    mW_hat = mW/(1.0 - beta1**t)
    mb_hat = mb/(1.0 - beta1**t)
    rW_hat = rW/(1.0 - beta2**t)
    rb_hat = rb/(1.0 - beta2**t)
    expected = [(gamma*mW_hat/(np.sqrt(rW_hat) + eps), gamma*mb_hat/(np.sqrt(rb_hat) + eps))]

    s = sum([np.sum(abs(Wc - We)) + np.sum(abs(bc - be)) for (Wc, bc), (We, be) in zip(computed, expected)])

    msg = f"Computed adjusted gradients {computed} do not equal expected adjusted gradients {expected}"
    assert s < 1e-12, msg

def test_NNR_fit():
    """
    Test that the fit method of the NeuralNetworkRegressor
    is able to approximate simple polynomials
    """
    rng = np.random.default_rng(2026)
    n_train = 10000
    n_test = 1000
    d = 1
    gamma = 0.1
    lam1 = 0.0
    lam2 = 0.0
    outputs = 1
    f = lambda x: x**2 - 3*x + 2
    x_train = np.linspace(-1, 1, n_train).reshape(n_train, -1)
    y_train = f(x_train)
    x_test = np.linspace(-1, 1, n_test).reshape(n_test, -1)
    expected = f(x_test)

    I = lambda x: x
    I_der = lambda x: np.ones_like(x)

    # Plain gradient descent
    net = NeuralNetworkRegressor(
        d,
        [50, 50, outputs],
        [ReLU, ReLU, I],
        [ReLU_der, ReLU_der, I_der],
        mse,
        mse_der,
        gamma=gamma,
        epochs=100,
        batch_size=1000,
        method="plain"
    )
    net.fit(x_train, y_train)
    computed = net.predict(x_test)

    err = np.mean((expected - computed)**2)

    msg = f"Plain gradient descent gave MSE {err} not sufficiently low"
    assert err < 1e-3, msg

    # RMSProp gradient descent
    gamma = 0.0003
    net = NeuralNetworkRegressor(
        d,
        [50, 50, outputs],
        [ReLU, ReLU, I],
        [ReLU_der, ReLU_der, I_der],
        mse,
        mse_der,
        gamma=gamma,
        epochs=100,
        batch_size=1000,
        method="rmsprop"
    )
    net.fit(x_train, y_train)
    computed = net.predict(x_test)

    err = np.mean((expected - computed)**2)

    msg = f"RMSProp gradient descent gave MSE {err} not sufficiently low"
    assert err < 1e-3, msg

    # Adam gradient descent
    gamma = 0.03
    net = NeuralNetworkRegressor(
        d,
        [50, 50, outputs],
        [ReLU, ReLU, I],
        [ReLU_der, ReLU_der, I_der],
        mse,
        mse_der,
        gamma=gamma,
        epochs=100,
        batch_size=1000,
        method="adam"
    )
    net.fit(x_train, y_train)
    computed = net.predict(x_test)

    err = np.mean((expected - computed)**2)

    msg = f"Adam gradient descent gave MSE {err} not sufficiently low"
    assert err < 1e-3, msg


if __name__ == "__main__":
    test_mse()
    test_mse_der()
    test_mse_Lp()
    test_cross_entropy()
    test_cross_entropy_Lp()
    test_cross_entropy_multiclass()
    test_ReLU()
    test_ReLU_der()
    test_LReLU()
    test_LReLU_der()
    test_sigmoid()
    test_sigmoid_der()
    test_softmax()
    test_NN_create_layers()
    test_NN_predict()
    test_NN_cost()
    test_NN__feed_forward_saver()
    test_NN_compute_gradient()
    test_update_weights()
    test_NNR_split_minibatches()
    test_NNR_optimiser_step()
    test_NNR_fit()
    print("All tests passed!")