import numpy as np
import sys
sys.path.append("../")
from models.neural_network import NeuralNetwork
from models.optimizer import Optimizer
from collections.abc import Callable
from sklearn.base import BaseEstimator, ClassifierMixin

class NeuralNetworkClassifier(Optimizer, BaseEstimator, ClassifierMixin):
    """
    Class representing a neural network with the scikit-learn estimator interface.

    Params
    ------
    network_input_size : int
        The number of inputs to the network
    layer_output_sizes : list[int]
        Ordered list of number of outputs of every layer
    classes : list
        List of classes
    activation_funcs : list[Callable]
        Ordered list of activation function of every layer
    activation_ders : list[Callable]
        Ordered list of derivatvie of activation function of every layer
    cost_fun : Callable
        Cost function for model evaluation
    cost_det : Callable
        Derivative of cost function
    gamma : float
        Learning rate (default = 0.03)
    lmbd1 : float
        L1 penalty (default = 0.0)
    lmbd2 : float
        L2 penalty (default = 0.0)
    epochs : int
        Number of gradient descent epochs (default = 1000)
    batch_size : int
        Number of batches per gradient descent epoch (default = 10)
    methods : str
        Gradient descent method. One of `plain`,
        `rmsprop`, `adam` (default = `plain`)
    rho : float
        Decay parameter used in `rmsprop` method (default=0.99)
    beta1 : float
        Momentum parameter used in `adam` method (default=0.9)
    beta2 : float
        Decay parameter used in `adam` method (default=0.999)
    eps : float
        Small constant used to avoid division by zero in
        `rmsprop` and `adam` methods (default=1e-8)
    rng : Generator
        Numpy random number generator (default = numpy.random.default_rng(2026))
    """
    def __init__(
            self,
            network_input_size: int,
            layer_output_sizes: list[int],
            classes: list,
            activation_funcs: list[Callable],
            activation_ders: list[Callable],
            cost_fun: Callable,
            cost_der: Callable,
            gamma: float = 0.03,
            lmbd1: float = 0.0,
            lmbd2: float = 0.0,
            epochs: int = 1000,
            batch_size: int = 10,
            method: str = "plain",
            rho: float = 0.99,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps = 1e-8,
            rng: np.random.Generator = np.random.default_rng(2026)
        ):
        # store the hyperparameters under the same names; no computation here
        self.nn_ = NeuralNetwork(
            network_input_size,
            layer_output_sizes,
            activation_funcs,
            activation_ders,
            cost_fun,
            cost_der,
            rng
        )
        self.classes = np.array(classes)
        self.gamma = gamma
        self.lmbd1 = lmbd1
        self.lmbd2 = lmbd2
        self.epochs = epochs
        self.batch_size = batch_size
        self.method = method
        self.rho = rho
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.rng = rng

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
        ) -> "NeuralNetworkClassifier":
        """
        Fits model to data

        Params
        ------
        X : np.ndarray
            Training data
        y : np.ndarray
            Target values

        Returns
        -------
        model : NeuralNetworkClassifier
            Fitted model
        """
        self.nn_.initialize_layers()
        self.train_stochastic(X, self._transform_target(y))
        return self

    def _transform_target(
            self,
            y: np.ndarray
        ) -> np.ndarray:
        """
        Transforms target for use in neural network

        Params
        ------
        y : np.ndarray
            Target labels

        Returns
        -------
        y_transformed : np.ndarray
            Transformed target
        """
        return ((self.classes).reshape(1, -1) == y.reshape(-1, 1)).astype(int)

    def classify(
            self,
            X: np.ndarray
        ) -> np.ndarray:
        """
        Classifies x

        Params
        ------
        x : ndarray
            Inputs

        Returns
        -------
        y : ndarray
            Predictions
        """
        pred = self.nn_.predict(X)
        return (self.classes[pred.argmax(axis=1)]).ravel()

    def predict(
            self,
            X: np.ndarray
        ) -> np.ndarray:
        """
        Predicts y as a function of x

        Params
        ------
        x : ndarray
            Inputs

        Returns
        -------
        y : ndarray
            Predictions
        """
        return self.nn_.predict(X)

    def accuracy(
            self,
            X: np.ndarray,
            y: np.ndarray
        ) -> np.ndarray:
        """
        Calculates sample accuracy

        Params
        ------
        x : ndarray
            Input values for prediction
        y : ndarray
            Target values

        Returns
        -------
        mse : float
            Mean square error
        """
        pred = self.classify(X)
        TP = np.sum((pred == y.ravel()))
        return TP/y.size

    def mse(
            self,
            X: np.ndarray,
            y: np.ndarray
        ) -> np.ndarray:
        """
        Calculates sample MSE

        Params
        ------
        x : ndarray
            Input values for prediction
        y : ndarray
            Target values

        Returns
        -------
        mse : float
            Mean square error
        """
        pred = self.predict(X)
        return np.mean((pred - self._transform_target(y))**2)