import numpy as np
import jax
import jax.numpy as jnp
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
        ):
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        # Simple feed forward pass
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)

        return a

    def cost(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        predicts = self.predict(inputs)
        return self.cost_fun(predicts, targets)

    def _feed_forward_saver(self, inputs):
        layer_inputs = []
        zs = []
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)

            zs.append(z)

        return layer_inputs, zs, a

    def compute_gradient(
            self,
            inputs: np.ndarray,
            targets: np.ndarray
        ) -> list[tuple[np.ndarray, np.ndarray]]:
        activation_funcs = self.activation_funcs
        layers = self.layers
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

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def update_weights(
            self,
            layer_grads: list[tuple[np.ndarray]],
            gamma: float
        ) -> None:
        layers = [
                    (W - gamma * dW, b - gamma * db)
                    for (W, b), (dW, db) in zip(self.layers, layer_grads)
                ]
        self.layers = layers

    # These last two methods are not needed in the project, but they can be nice to have!
    # The first one has a layers parameter so that you can use jax.grad on it.
    def jax_compliant_predict(
            self,
            layers: list[tuple[np.ndarray]],
            inputs: np.ndarray
        ) -> np.ndarray:
        a = inputs
        for (W, b), activation_func in zip(layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)

        return a

    def jax_compliant_cost(
            self,
            layers: list[tuple[np.ndarray]],
            inputs: np.ndarray,
            targets: np.ndarray
        ) -> float:
        predicts = self.jax_compliant_predict(layers, inputs)
        return self.cost_fun(predicts, targets)

    def jax_gradient(
            self,
            inputs: np.ndarray,
            targets: np.ndarray
        ) -> np.ndarray:
        cost_grad = jax.grad(self.jax_compliant_cost, argnums=0)
        layer_grads_jax = cost_grad(
            self.layers,
            inputs,
            self.activation_funcs,
            targets
        )

        return layer_grads_jax