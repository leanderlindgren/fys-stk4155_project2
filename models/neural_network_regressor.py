import numpy as np
import sys
sys.path.append("../")
from models.neural_network import NeuralNetwork
from collections.abc import Callable
from sklearn.base import BaseEstimator, RegressorMixin

class NeuralNetworkRegressor(BaseEstimator, RegressorMixin):
    """
    Class representing a neural network with the scikit-learn estimator interface.

    Params
    ------
    network_input_size : int
        The number of inputs to the network
    layer_output_sizes : list[int]
        Ordered list of number of outputs of every layer
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
        ) -> "NeuralNetworkRegressor":
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
        model : NeuralNetworkRegressor
            Fitted model
        """
        self.nn_.initialize_layers()
        self.train_stochastic(X, y)
        return self

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
        return np.mean((pred - y)**2)

    def optimiser_step(
            self,
            layer_grads: list[tuple[np.ndarray, np.ndarray]],
            state: dict[str, float],
            t: int,
            method: str = "plain",
            rho: float = 0.99,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps = 1e-8
        ):
        """
        Applies a step of gradient descent to model stored as attribute.

        Params
        -----
        layer_grads : list[tuple[ndarray, ndarray]]
            Ordered list of gradients of weights and biases for each layer
        state : dict[str, float]
            Maintains momentum and RMS coefficients
        t : int
            Current epoch number
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

        Returns
        -------
        adjusted_grads : list[tuple[ndarray, ndarray]]
            Ordered list of adjusted gradients for each layer
        """
        gamma = self.gamma
        adjusted_grads = []
        # Having the switch statement inside the loop will
        # improve readability at negligible cost to performance
        for i, (dW, db) in enumerate(layer_grads):
            match method:
                case "plain":
                    adjusted_grads.append(
                        (gamma*dW, gamma*db)
                    )

                case "rmsprop":
                    rW = rho*state.get(f"rW_{i+1}", 0.0) + (1.0 - rho)*dW**2
                    rb = rho*state.get(f"rb_{i+1}", 0.0) + (1.0 - rho)*db**2

                    state[f"rW_{i+1}"] = rW
                    state[f"rb_{i+1}"] = rb

                    adjusted_grads.append(
                        (
                            gamma*dW/(np.sqrt(rW) + eps),
                            gamma*db/(np.sqrt(rb) + eps)
                        )
                    )

                case "adam":
                    mW = beta1*state.get(f"mW_{i+1}", 0.0) + (1.0 - beta1)*dW
                    mb = beta1*state.get(f"mb_{i+1}", 0.0) + (1.0 - beta1)*db
                    rW = beta2*state.get(f"rW_{i+1}", 0.0) + (1.0 - beta2)*dW**2
                    rb = beta2*state.get(f"rb_{i+1}", 0.0) + (1.0 - beta2)*db**2

                    state[f"mW_{i+1}"] = mW
                    state[f"mb_{i+1}"] = mb
                    state[f"rW_{i+1}"] = rW
                    state[f"rb_{i+1}"] = rb

                    mW_hat = mW/(1.0 - beta1**t)
                    mb_hat = mb/(1.0 - beta1**t)
                    rW_hat = rW/(1.0 - beta2**t)
                    rb_hat = rb/(1.0 - beta2**t)

                    adjusted_grads.append(
                        (
                            gamma*mW_hat/(np.sqrt(rW_hat) + eps),
                            gamma*mb_hat/(np.sqrt(rb_hat) + eps)
                        )
                    )

                case _:
                    raise ValueError(f"Unknown method {method}")

        return adjusted_grads

    def split_minibatches(
            self,
            inputs: np.ndarray,
            targets: np.ndarray
        ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Helper function for splitting data into minibatches

        Params
        ------
        inputs : np.ndarray
            Training data
        targets : np.ndarray
            Target values

        Returns
        -------
        minibatches : list[tuple[ndarray, ndarray]]
            List of tuples of inputs and outputs
        """
        n = len(targets)
        # Resample data in case n is not divisible by batch size
        if n%self.batch_size:
            num_resample = self.batch_size - n%self.batch_size
            resample_idx = self.rng.permutation(n)
            inputs = np.append(inputs, inputs[resample_idx[:num_resample]], axis=0)
            targets = np.append(targets, targets[resample_idx[:num_resample]], axis=0)
        else:
            num_resample = 0

        n += num_resample

        shuffled_idx = self.rng.permutation(n)
        inputs_shuffled = inputs[shuffled_idx]
        targets_shuffled = targets[shuffled_idx]
        input_batches = np.split(inputs_shuffled, n//self.batch_size, axis=0)
        target_batches = np.split(targets_shuffled, n//self.batch_size, axis=0)

        batches = []
        for input_mini, target_mini in zip(input_batches, target_batches):
            batches.append((input_mini, target_mini))

        return batches

    def train_stochastic(
            self,
            inputs: np.ndarray,
            targets: np.ndarray
        ) -> None:
        state = {}
        for i in range(self.epochs):
            print(f"Epoch: {i+1}/{self.epochs}\r", end="")
            minibatches = self.split_minibatches(inputs, targets)
            n = (inputs.shape[0] + self.batch_size
                 - inputs.shape[0]%self.batch_size)
            for (inputs_mini, targets_mini) in minibatches:
                layer_grads = self.nn_.compute_gradient(
                    inputs_mini,
                    targets_mini,
                    self.lmbd1,
                    self.lmbd2
                )
                adjusted_grads = self.optimiser_step(
                    layer_grads,
                    state,
                    t=i+1,
                    method=self.method,
                    rho=self.rho,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    eps=self.eps
                )
                self.nn_.update_weights(adjusted_grads, 1.0)

        print(" "*len(f"\rEpoch: {self.epochs}/{self.epochs}"), end="\r")