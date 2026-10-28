import numpy as np

def runge(x: np.ndarray) -> np.ndarray:
    """
    Runge function `1/(1+25x^2)`

    Params
    ------
    x : ndarray
        Values to evaluate function at
    
    Returns
    -------
    y : ndarray
        Values of Runge function at **x**
    """
    return 1/(1 + 25*x**2)