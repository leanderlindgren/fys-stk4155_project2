import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from cost_functions import mse, mse_Lp, cross_entropy, \
        cross_entropy_Lp, cross_entropy_multiclass
from activation_functions import ReLU, ReLU_der, LReLU, LReLU_der, sigmoid, \
        sigmoid_der, softmax, ReLU_jax, LReLU_jax, sigmoid_jax, softmax_jax

def test_mse():
    """
    Test that the mse function returns correct MSE
    """
    y = np.array([0.8, 0.2, -1.9, 1.4, 0.6, -0.3, -0.3, 0.3, -0.3, -0.2])
    y_pred = np.array([0.7, 0.5, -0.1, -0.1, 0.2, -0.6, -0.4, 0.5, -0.1, -1.4])
    expected = 0.737
    computed = mse(y_pred, y)

    msg = f"Computed MSE {computed} does not equal expected MSE {expected}"
    assert abs(expected - computed) < 1e-12, msg

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

    msg = f"Computed ReLU_der(z) {computed} does not equal expected ReLU_der(z) {expected}"
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

if __name__ == "__main__":
    test_mse()
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