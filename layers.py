import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

def apply_dense_layer(w, b, x, activation=lambda x:x):
    '''Apply one dense layer with weight w, bias b, and an activation function'''
    return activation(jnp.dot(w, x) + b)

def global_conv_layer(num_channels, min_xi, max_xi, dx, grids)
    """Global convolution layer, Eq. S9 in the paper."""
    assert num_channels > 0
    self.displacements = tf.expand_dims(grids, axis=1) - tf.expand_dims(grids, axis=0)
    eta_init = tf.keras.initializers.RandomNormal(0, 0.0001)
    self.eta = tf.Variable(
        initial_value=eta_init(shape=(num_channels,)), dtype="float64",
        trainable=True)

    def _exponential_displacements_func(self, width):
        """Exponential function.
        Args:
            width: Float, parameter of exponential function.
        Returns:
            Float tensor with same shape as self.displacements.
        """

        return jnp.exp(-jnp.abs(self.displacements) / width) / (2*width)

    def call(self, inputs):
        """Args:
            inputs: n(x) Float tensor, shape (batch_size, num_grids).
        Returns:
            Float tensor, shape (batch_size, num_grids, num_channels)."""
        
        widths= min_xi + max_xi - min_xi * tf.keras.activations.sigmoid(eta)
        kernels = tf.vectorized_map(self._exponential_displacements_func, widths) #(num_channels, grid_size, grid_size)
        kernels = tf.transpose(kernels, [1,2,0]) #(grid_size, grid_size, num_channels)

        return tf.tensordot(inputs, kernels, axes=(1, 0)) * self.dx

    return call