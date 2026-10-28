import numpy as np
import jax.numpy as jnp

def mse_jax(predict: np.ndarray, target: np.ndarray) -> float:
    """
    Mean square error of predictions against target.

    Params
    ------
    predict : ndarray
        Predictions
    target : ndarray
        Targets

    Returns
    -------
    mse : float
        Mean square error

    Notes
    -----
    This function should only be used for calculating gradients with jax,
    as it can be slower than the version that uses numpy instead.
    """
    return jnp.mean((predict - target)**2)

def mse(predict: np.ndarray, target: np.ndarray) -> float:
    """
    Mean square error of predictions against target.

    Params
    ------
    predict : ndarray
        Predictions
    target : ndarray
        Targets

    Returns
    -------
    mse : float
        Mean square error
    """
    return np.mean((predict - target)**2)

def mse_der(predict: np.ndarray, target: np.ndarray) -> float:
    """
    Derivative of mean square error.

    Params
    ------
    predict : ndarray
        Predictions
    target : ndarray
        Targets

    Returns
    -------
    mse_der : float
        Derivative of mean square error
    """
    return (2/predict.size)*(predict - target)

def mse_Lp(
        predict: np.ndarray,
        target: np.ndarray,
        layers: list[tuple[np.ndarray]],
        lam: float,
        order: int
    ) -> float:
    """
    Sum of mean square error of predictions against
    target and an L1 or L2 penalty term.

    Params
    ------
    predict : ndarray
        Predictions
    target : ndarray
        Targets
    layers : list
        Weights and biases of layers
    lam : float
        Strength of penalty term
    order : int
        Order of penalty term

    Returns
    -------
    mse : float
        Mean square error + penalty

    Notes
    -----
    lam parameter should be greater than 0,
    and order parameter should be either 1 or 2.
    """
    if lam < 0:
        raise ValueError("Lambda must be greater than or equal to 0")
    if not isinstance(order, int):
        raise ValueError(f"Order must be an integer, not a {type(order)}")
    if order < 1:
        raise ValueError(f"Order must be an integer greater than 0")
    if order > 2:
        raise ValueError(f"Order should be either 1 or 2")

    return mse(predict, target) + lam*_norm(layers, order)

def _norm(layers: list[tuple[np.ndarray]], order: int) -> float:
    """
    L1 and L2 norm helper function.

    Params
    ------
    layers : list
        Weights and biases of layers
    order : int
        Order of penalty term

    Returns
    -------
    Lp : float
        Lp-norm of weights
    """
    penalty = 0
    for (W, b) in layers:
        penalty += np.sum(abs(W)**order) + np.sum(abs(b)**order)
    return np.sqrt(penalty)

def cross_entropy(predict: np.ndarray, target: np.ndarray) -> float:
    """
    Binary cross entropy/log loss function.

    Params
    ------
    predict : ndarray
        Predictions
    target : ndarray
        Targets

    Returns
    -------
    mse : float
        Cross entropy
    """
    return np.sum(-target * np.log(predict + 1e-12))

def cross_entropy_Lp(
        predict: np.ndarray,
        target: np.ndarray,
        layers: list[tuple[np.ndarray]],
        lam: float,
        order: int
    ) -> float:
    """
    Binary cross entropy/log loss function.

    Params
    ------
    predict : ndarray
        Predictions
    target : ndarray
        Targets
    layers : list
        Weights and biases of layers
    lam : float
        Strength of penalty term
    order : int
        Order of penalty term

    Returns
    -------
    cross_entropy : float
        Cross entropy + penalty

    Notes
    -----
    lam parameter should be greater than 0,
    and order parameter should be either 1 or 2.
    """
    if lam < 0:
        raise ValueError("Lambda must be greater than or equal to 0")
    if not isinstance(order, int):
        raise ValueError(f"Order must be an integer, not a {type(order)}")
    if order < 1:
        raise ValueError(f"Order must be an integer greater than 0")
    if order > 2:
        raise ValueError(f"Order should be either 1 or 2")

    return cross_entropy(predict, target) + lam*_norm(layers, order)

def cross_entropy_multiclass(predict: np.ndarray, target: np.ndarray) -> float:
    """
    Multiclass cross entropy/log loss function.

    Params
    ------
    predict : ndarray
        Predictions
    target : ndarray
        Targets

    Returns
    -------
    cross_entropy : float
        Cross entropy
    """
    return np.mean(np.sum(-target*np.log(predict + 1e-12), axis=1))