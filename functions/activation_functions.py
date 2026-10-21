# Defining some activation functions
def ReLU(z):
    return jnp.where(z > 0, z, 0.0)


def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))

