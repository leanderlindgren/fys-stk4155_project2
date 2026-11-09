import sys
sys.path.append("../")
import numpy as np
from functions.linear_regression import Ridge_closed_form
from functions.cost_functions import mse, mse_der
from models.ols import OLS
from models.optimizer import Optimizer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from models.neural_network import NeuralNetwork

class Lasso(OLS):
    def __init__(
            self,
            lmbd: float,
            degree: int,
            scale_mean: bool = True,
            scale_sd: bool = True,
            gamma: float = 0.03,
            epochs: int = 1000,
            batch_size: int = 10,
            method: str = "plain",
            rho: float = 0.99,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps = 1e-8,
            rng: np.random.Generator = np.random.default_rng(2026)
        ):
        self.scaler = StandardScaler(with_mean=scale_mean, with_std=scale_sd)
        self.lmbd = lmbd
        self.degree = degree
        self.gamma = gamma
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
            x: np.ndarray,
            y: np.ndarray
        ) -> "Lasso":
        self.pipeline = make_pipeline(
            PolynomialFeatures(self.degree, include_bias=False),
            self.scaler
        )
        X_scaled = self.pipeline.fit_transform(x.reshape(-1, 1))
        self.intercept_ = y.mean()
        self.coef_ = self.rng.normal(
            scale=np.sqrt(2.0/self.degree),
            size=(self.degree)
        )
        y_centered = y - y.mean()
        self.train_stochastic(X_scaled, y_centered)
        return self

    def optimiser_step(
            self,
            grad: np.ndarray,
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
        grads : ndarray
            Gradients of parameters
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
        grads : ndarray
            Ordered list of adjusted gradients for each layer
        """
        gamma = self.gamma
        match method:
            case "plain":
                adjusted_grad = gamma*grad

            case "rmsprop":
                r = rho*state.get("r", 0.0) + (1.0 - rho)*grad**2

                state["r"] = r

                adjusted_grad = gamma*grad/(np.sqrt(r) + eps)

            case "adam":
                m = beta1*state.get("m", 0.0) + (1.0 - beta1)*grad
                r = beta2*state.get("r", 0.0) + (1.0 - beta2)*grad**2

                state["m"] = m
                state["r"] = r

                m_hat = m/(1.0 - beta1**t)
                r_hat = r/(1.0 - beta2**t)

                adjusted_grad = gamma*m_hat/(np.sqrt(r_hat) + eps)

            case _:
                raise ValueError(f"Unknown method {method}")

        return adjusted_grad

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
            if self.batch_size < inputs.shape[0]:
                minibatches = self.split_minibatches(inputs, targets)
                for (inputs_mini, targets_mini) in minibatches:
                    grad = (1/inputs_mini.shape[0])*inputs_mini.T@(inputs_mini@self.coef_ - targets_mini.ravel())
                    grad += self.lmbd*np.sign(self.coef_)
                    adjusted_grads = self.optimiser_step(
                        grad,
                        state,
                        t=i+1,
                        method=self.method,
                        rho=self.rho,
                        beta1=self.beta1,
                        beta2=self.beta2,
                        eps=self.eps
                    )
                    self.coef_ -= adjusted_grads
            elif self.batch_size == inputs.shape[0]:
                grad = (1/inputs.shape[0])*inputs.T@(inputs@self.coef_ - targets.ravel())
                grad += self.lmbd*np.sign(self.coef_)
                adjusted_grads = self.optimiser_step(
                    grad,
                    state,
                    t=i+1,
                    method=self.method,
                    rho=self.rho,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    eps=self.eps
                )
                self.coef_ -= adjusted_grads
            else:
                raise ValueError(f"Batch size {self.batch_size} is larger than sample size {inputs.shape[0]}")

        print(" "*len(f"\rEpoch: {self.epochs}/{self.epochs}"), end="\r")