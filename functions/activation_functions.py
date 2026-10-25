import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np

def ReLU_jax(z: jnp.ndarray) -> jnp.ndarray:
    """
    ReLU activation function.

    Params
    ------
    z : ndarray
        Unactivated layer output

    Returns
    -------
    a : ndarray
        Activated layer output

    Notes
    -----
    This function should only be used for calculating gradients with jax,
    as it can be slower than the version that uses numpy instead.
    """
    return jnp.where(z > 0, z, 0.0)

def LReLU_jax(z: np.ndarray, alpha: float) -> np.ndarray:
    """
    Leaky ReLU activation function.

    Params
    ------
    z : ndarray
        Unactivated layer output

    alpha : float
        Leaking parameter

    Returns
    -------
    a : ndarray
        Activated layer output

    Notes
    -----
    This function should only be used for calculating gradients with jax,
    as it can be slower than the version that uses numpy instead.
    """
    positive = z >= 0
    negative = ~positive

    a = jnp.empty_like(z, dtype=float)
    a = a.at[positive].set(z[positive]) + a.at[negative].set(alpha*z[negative])
    return a

def sigmoid_jax(z: jnp.ndarray) -> jnp.ndarray:
    """
    Sigmoid activation function.

    Params
    ------
    z : ndarray
        Unactivated layer output

    Returns
    -------
    a : ndarray
        Activated layer output

    Notes
    -----
    This function should only be used for calculating gradients with jax,
    as it can be slower than the version that uses numpy instead.
    """
    # We use this method to avoid instability for large negative values
    negative = z < 0
    return abs(negative - (1 / (1 + jnp.exp(-abs(z)))))

def softmax_jax(z: jnp.ndarray) -> jnp.ndarray:
    """
    Softmax activation function.

    Params
    ------
    z : ndarray
        Unactivated layer output

    Returns
    -------
    a : ndarray
        Activated layer output

    Notes
    -----
    This function should only be used for calculating gradients with jax,
    as it can be slower than the version that uses numpy instead.
    """
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)

def ReLU(z: np.ndarray) -> np.ndarray:
    """
    ReLU activation function.

    Params
    ------
    z : ndarray
        Unactivated layer output

    Returns
    -------
    a : ndarray
        Activated layer output

    Notes
    -----
    This function should not be used for computing gradients with jax,
    where ReLU_jax should be used instead.
    """
    return np.where(z > 0, z, 0.0)

def ReLU_der(z: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU activation function.

    Params
    ------
    z : ndarray
        Unactivated layer output

    Returns
    -------
    da_dz : ndarray
        Derivative of activated layer output
    """
    return np.where(z > 0, 1.0, 0.0)

def LReLU(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Leaky ReLU activation function.

    Params
    ------
    z : ndarray
        Unactivated layer output

    alpha : float
        Leaking parameter (default=0.01)

    Returns
    -------
    a : ndarray
        Activated layer output

    Notes
    -----
    This function should not be used for computing gradients with jax,
    where LReLU_jax should be used instead.
    """
    positive = z >= 0
    negative = ~positive

    a = np.empty_like(z, dtype=float)
    a[positive] = z[positive]
    a[negative] = alpha*z[negative]
    return a

def LReLU_der(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Derivative of leaky ReLU activation function.

    Params
    ------
    z : ndarray
        Unactivated layer output

    alpha : float
        Leaking parameter (default=0.01)

    Returns
    -------
    da_dz : ndarray
        Derivative of activated layer output
    """
    positive = z >= 0
    negative = ~positive

    a = np.empty_like(z, dtype=float)
    a[positive] = 1.0
    a[negative] = -alpha
    return a

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.

    Params
    ------
    z : ndarray
        Unactivated layer output

    Returns
    -------
    a : ndarray
        Activated layer output

    Notes
    -----
    This function should not be used for computing gradients with jax,
    where sigmoid_jax should be used instead.
    """
    # We use this method to avoid instability for large negative values
    # A bit slower when z has few values,
    # but almost as fast when z has many values (from limited testing)
    negative = z < 0
    return abs(negative - (1 / (1 + np.exp(-abs(z)))))

def sigmoid_der(z: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid activation function.

    Params
    ------
    z : ndarray
        Unactivated layer output

    Returns
    -------
    da_dz : ndarray
        Derivative of activated layer output
    """
    # We use this method to avoid instability for large negative values
    exp = np.exp(-abs(z))
    return exp/(1+exp)**2

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax activation function.

    Params
    ------
    z : ndarray
        Unactivated layer output

    Returns
    -------
    a : ndarray
        Activated layer output

    Notes
    -----
    This function should not be used for computing gradients with jax,
    where softmax_jax should be used instead.
    """
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)