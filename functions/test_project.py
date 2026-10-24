import numpy as np
from cost_functions import mse, mse_Lp, cross_entropy, \
        cross_entropy_Lp, cross_entropy_multiclass

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

if __name__ == "__main__":
    test_mse()
    test_mse_Lp()
    test_cross_entropy()
    test_cross_entropy_Lp()
    test_cross_entropy_multiclass()