"""A tensorflow/keras implementation of Fig S3(a) in Li et al., “Kohn-Sham equations as regularizer:
        Building prior knowledge into machine-learned physics,” arXiv:2009.08551 (2020),"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from jax_dft import datasets
from jax_dft import utils

tf.keras.backend.set_floatx('float64')

class ExponentialGlobalConv(tf.keras.layers.Layer):
    """Global convolution layer, Eq. S9 in the paper."""

    def __init__(self, num_channels, min_xi, max_xi, dx, grids):
        super().__init__()
        assert num_channels > 0

        self.num_channels = num_channels
        self.min_xi = min_xi
        self.max_xi = max_xi
        self.dx = dx #grid spacing
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

        return tf.math.exp(-tf.math.abs(self.displacements) / width) / (2*width)

    def call(self, inputs):
        """Args:
            inputs: n(x) Float tensor, shape (batch_size, num_grids).
        Returns:
            Float tensor, shape (batch_size, num_grids, num_channels)."""
        
        widths= self.min_xi + (self.max_xi - self.min_xi) * tf.keras.activations.sigmoid(self.eta)
        kernels = tf.vectorized_map(self._exponential_displacements_func, widths) #(num_channels, grid_size, grid_size)
        kernels = tf.transpose(kernels, [1,2,0]) #(grid_size, grid_size, num_channels)
        
        #print(inputs.shape, kernels.shape)
        return tf.tensordot(inputs, kernels, axes=(1, 0)) * self.dx
        #print(result.shape)
        #return result

data = datasets.Dataset(path='data/h2/', num_grids=513)

'''model = tf.keras.Sequential([
    ExponentialGlobalConv(16,min_xi=0.1,max_xi=2.385345,dx=0.08,grids=data.grids),
    tf.keras.layers.Conv1D(
        filters=16, kernel_size=3, strides=1, padding='same',
        activation='swish', use_bias=False),
    tf.keras.layers.Conv1D(
        filters=16, kernel_size=3, strides=1, padding='same',
        activation='swish', use_bias=False),
    tf.keras.layers.Conv1D(
        filters=128, kernel_size=3, strides=1, padding='same',
        activation='swish', use_bias=False),
    tf.keras.layers.MaxPool1D(
        pool_size=2, strides=None, padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='swish'),
    tf.keras.layers.Dense(1)
])'''

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='sigmoid'),
    tf.keras.layers.Dense(1)
])

# "swish" AKA SiLU activation function = x*sigmoid(x)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                loss=tf.keras.losses.MeanSquaredError())

train_distances = [128, 384]
mask = np.isin(data.distances_x100, train_distances)
#model.fit(data.densities[mask,:], data.total_energies[mask], epochs = 100)
model.fit(data.densities, data.total_energies, epochs = 300)
print(model.summary())
#model.save("h2_ml_model")
train_energies = tf.reshape(model(data.densities), -1)

nuclear_energy = utils.get_nuclear_interaction_energy_batch(
    data.locations,
    data.nuclear_charges,
    interaction_fn=utils.exponential_coulomb)

plt.plot(data.distances, data.total_energies+nuclear_energy, linestyle='dashed', color='black', label='Exact')
plt.plot(data.distances, train_energies+nuclear_energy, color='purple', label='Network trained on distance=1.28 and 3.84')
#plt.plot(data.distances[mask], (train_energies+nuclear_energy)[mask], marker='D', linestyle='None')

plt.xlabel('Distance')
plt.ylabel('Total Energy + E_nn')
plt.legend()
#plt.savefig("h2_NN_prediction.png")
plt.show()
