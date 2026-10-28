import numpy as np
import jax
import jax.numpy as jnp
from functions.activation_functions import ReLU
jax.config.update("jax_enable_x64", True)

class NeuralNetwork:
    def __init__(
            self,
            network_input_size,
            layer_output_sizes,
            activation_funcs,
            activation_ders,
            cost_fun,
            cost_der,
            rng = np.random.default_rng(2026)
        ):
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.rng = rng

    def create_layers(
            self,
            network_input_size: int,
            layer_output_sizes: list[int]
        ) -> list[tuple[np.ndarray]]:
        """
        Creates randomly initialized layers

        Params
        ------
        network_input_size : int
            Number of inputs of the network
        layer_output_sizes : list[int]
            Ordered list of number of outputs per layer
        
        Returns
        -------
        layers : list[tuple[ndarray]]
            Ordered list containing a tuple of the
            weights and biases of each layer
        """
        layers = []

        i_size = network_input_size
        for (layer_output_size, activation_func) in zip(layer_output_sizes, self.activation_funcs):
            scale = np.sqrt(2.0 / i_size) if activation_func == ReLU else np.sqrt(1.0 / i_size)
            W = self.rng.normal(scale=scale, size=(i_size, layer_output_size))
            b = np.zeros(layer_output_size) + 0.01
            layers.append((W, b))

            i_size = layer_output_size

        return layers

    def predict(
            self,
            x: np.ndarray
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
        # Simple feed forward pass
        a = x
        for (W, b), activation_func in zip(self.layers_, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)

        return a

    def cost(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            lam1: float = 0.0,
            lam2: float = 0.0
        ) -> float:
        """
        Calculates cost/loss specified by cost_fun attribute

        Params
        ------
        inputs : ndarray
            Inputs used to calculate predictions
        targets : ndarray
            Targets to compare against predictions
        lam1 : float
            L1 penalty parameter
        lam2 : float
            L2 penalty parameter
        
        Returns
        -------
        cost : float
            Cost/loss
        """
        predicts = self.predict(inputs)
        cost = self.cost_fun(predicts, targets)
        if lam1 > 0.0:
            l1_penalty = lam1 * sum(np.sum(abs(W)) for (W, _) in self.layers_)
            cost += l1_penalty
        if lam2 > 0.0:
            l2_penalty = lam2 * sum(np.sum(W**2) for (W, _) in self.layers_)
            cost += l2_penalty
        return cost

    def _feed_forward_saver(
            self,
            inputs: np.ndarray
        ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """
        Calculates predictions, but also saves the
        inputs and unactivated outputs for every layer

        Params
        ------
        inputs : ndarray
            Inputs used to calculate predictions
        
        Returns
        -------
        layer_inputs : list[ndarray]
            Inputs of every layer
        zs : list[ndarray]
            Unactivated outputs of every layer
        predicts : ndarray
            Predictions
        """
        layer_inputs = []
        zs = []
        a = inputs
        for (W, b), activation_func in zip(self.layers_, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)

            zs.append(z)

        return layer_inputs, zs, a

    def compute_gradient(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            lam1: float = 0.0,
            lam2: float = 0.0
        ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Calculates gradients with respect to weights and biases for every layer

        Params
        ------
        inputs : ndarray
            Inputs for calculating gradient and
            calculating predictions for cost function
        targets : ndarray
            Targets for cost function
        lam1 : float
            L1 penalty parameter
        lam2 : float
            L2 penalty parameter
        
        Returns
        -------
        layer_grads : list[tuple[ndarray, ndarray]]
            Gradients with respect to weights and biases for every layer
        """
        activation_funcs = self.activation_funcs
        layers = self.layers_
        layer_inputs, zs, predicts = self._feed_forward_saver(inputs, layers, activation_funcs)

        layer_grads = [() for layer in layers]

        # We loop over the layers, from the last to the first
        for i in reversed(range(len(layers))):
            layer_input = layer_inputs[i]
            z = zs[i]
            activation_der = self.activation_ders[i]

            if i == len(layers) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                if activation_der == None:
                    dC_dz = predicts - targets
                else:
                    dC_da = self.cost_der(predicts, targets)
                    dC_dz = (dC_da * activation_der(z).T).T
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W, b) = layers[i + 1]
                dC_da = dC_dz @ W.T
                dC_dz = dC_da * activation_der(z)

            dC_dW = layer_input.T @ dC_dz
            dC_db = np.sum(dC_dz, axis=0)

            # Apply L1 and L2 penalties
            W = layers[i][0]
            if lam1 > 0.0:
                dC_dW += lam1*np.sign(W)
            if lam2 > 0.0:
                dC_dW += lam2*2*W
            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def update_weights(
            self,
            layer_grads: list[tuple[np.ndarray]],
            gamma: float
        ) -> None:
        """
        Updates weights and biases by gradients

        Params
        ------
        layer_grads : list[tuple[ndarray]]
            Ordered list of gradients of the weights and biases
        gamma : float
            Learning rate
        """
        layers = [
                    (W - gamma * dW, b - gamma * db)
                    for (W, b), (dW, db) in zip(self.layers_, layer_grads)
                ]
        self.layers_ = layers

    # These last two methods are not needed in the project, but they can be nice to have!
    # The first one has a layers parameter so that you can use jax.grad on it.
    def jax_compliant_predict(
            self,
            layers: list[tuple[np.ndarray]],
            inputs: np.ndarray
        ) -> np.ndarray:
        """
        Calculates predictions

        Params
        ------
        layers : list[tuple[ndarray]]
            Layers of the network
        inputs : ndarray
            Inputs used to calculate predictions
        
        Returns
        -------
        predicts : ndarray
            Predictions
        
        Notes
        -----
        Should be used with JAX
        """
        a = inputs
        for (W, b), activation_func in zip(layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)

        return a

    def jax_compliant_cost(
            self,
            layers: list[tuple[np.ndarray]],
            inputs: np.ndarray,
            targets: np.ndarray,
            lam1: float,
            lam2: float
        ) -> float:
        """
        Calculates cost/loss specified by cost_fun attribute

        Params
        ------
        layers : list[tuple[ndarray]]
            Layers of the network
        inputs : ndarray
            Inputs used to calculate predictions
        targets : ndarray
            Targets to compare against predictions
        lam1 : float
            L1 penalty parameter
        lam2 : float
            L2 penalty parameter
        
        Returns
        -------
        cost : float
            Cost/loss
        
        Notes
        -----
        Should be used with JAX
        """
        predicts = self.jax_compliant_predict(layers, inputs)
        cost = self.cost_fun(predicts, targets)
        if lam1 > 0.0:
            l1_penalty = lam1 * sum(jnp.sum(abs(W)) for (W, _) in self.layers_)
            cost += l1_penalty
        if lam2 > 0.0:
            l2_penalty = lam2 * sum(jnp.sum(W**2) for (W, _) in self.layers_)
            cost += l2_penalty
        return cost

    def jax_gradient(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            lam1: float = 0.0,
            lam2: float = 0.0
        ) -> np.ndarray:
        """
        Calculates gradients with respect to
        weights and biases for every layer using JAX

        Params
        ------
        inputs : ndarray
            Inputs for calculating gradient and
            calculating predictions for cost function
        targets : ndarray
            Targets for cost function
        lam1 : float
            L1 penalty parameter
        lam2 : float
            L2 penalty parameter
        
        Returns
        -------
        layer_grads : list[tuple[ndarray, ndarray]]
            Gradients with respect to weights and biases for every layer
        """
        cost_grad = jax.grad(self.jax_compliant_cost, argnums=0)
        layer_grads_jax = cost_grad(
            self.layers_,
            inputs,
            targets,
            lam1,
            lam2
        )

        return layer_grads_jax