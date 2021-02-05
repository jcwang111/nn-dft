import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.nn import sigmoid
from jax import lax
from jax.nn.initializers import he_normal, glorot_normal

def get_dense_layer(m, n, activation):
    def init_dense(key, scale=1E-4):
        '''Generate weights and biases for a m->n layer'''
        w_key, b_key = random.split(key)
        return glorot_normal()(w_key, (n,m)), scale*random.normal(b_key, (n,))

    def predict(params, x):
        '''Apply one dense layer with weight w, bias b, and an activation function'''
        w, b = params[0], params[1]
        return activation(jnp.dot(x, w.T) + b)

    return init_dense, predict

def get_global_conv_layer(num_channels, min_xi, max_xi, dx, grids):
    """Global convolution layer, Eq. S9 in the paper."""
    displacements = grids[:,None] - grids[None,:]

    def _exponential_displacements_func(width):
        """Exponential function.
        Args:
            width: Float, parameter of exponential function.
        Returns:
            Float array with same shape as self.displacements.
        """

        return jnp.exp(-jnp.abs(displacements) / width) / (2*width)

    def init_params(key, scale=1E-4):
        """Initialize eta values"""
        return scale*random.normal(key, (num_channels,))

    def predict(params, inputs):
        """Args:
            params: shape (8,) array, eta values
            inputs: n(x) Float tensor, shape (batch_size, num_grids).
        Returns:
            Float array, shape (batch_size, num_grids, num_channels)."""
        
        widths = min_xi + max_xi - min_xi * sigmoid(params) #(num_channels,)
        kernels = vmap(_exponential_displacements_func, out_axes=2)(widths) #(grid_size, grid_size, num_channels)
        return jnp.tensordot(inputs, kernels, axes=(1, 0)) * dx #(batch_size, 1, num_grids, num_channels)

    return init_params, predict

def get_conv_layer(window_size, in_channels, out_channels, activation):
    """1D Convolution with a window size of 3."""

    def init_weights(key, scale=1):
        """Initialize eta values"""
        return scale*he_normal()(key, (window_size, in_channels, out_channels))

    def predict(kernel, inputs):
        """Args:
            kernel: params input, shape (window_size, in_channels, out_channels)
            inputs: channel inputs, shape (batch_size, num_grids, in_channels).
        Returns:
            Float array, shape (batch_size, num_grids, out_channels)."""

        dn = lax.conv_dimension_numbers(inputs.shape, kernel.shape, ('NWC', 'WIO', 'NWC'))
        out = lax.conv_general_dilated(inputs, kernel,
                                        window_strides=(1,), padding='SAME', dimension_numbers=dn) 
        return activation(out)

    return init_weights, predict

def get_maxpool2_flatten_dense(m, n, activation):
    """Maxpool over every block of 2 and flatten, then apply dense layer"""
    def init_dense(key, scale=1E-4):
        '''Generate weights and biases for a m->n layer'''
        w_key, b_key = random.split(key)
        return glorot_normal()(w_key, (n,m)), scale*random.normal(b_key, (n,))

    def predict(params, x):
        '''Apply one dense layer with weight w, bias b, and an activation function'''
        x = jnp.maximum(x[:,0:-1:2,:], x[:,1::2,:]).reshape(x.shape[0], -1) #Maxpool 2
        w, b = params[0], params[1]
        return activation(jnp.dot(x, w.T) + b)

    return init_dense, predict