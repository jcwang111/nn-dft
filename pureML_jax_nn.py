"""A reimplementation, also using JAX, of Fig S3(a) in Li et al., “Kohn-Sham equations as regularizer:
        Building prior knowledge into machine-learned physics,” arXiv:2009.08551 (2020),"""

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.nn import sigmoid, relu, softplus, silu, elu
from jax.scipy import optimize

import matplotlib.pyplot as plt
from jax_dft import datasets
from jax_dft import utils

from pureML_layers import *

#Retrieve data
data = datasets.Dataset(path='data/h2/', num_grids=513)
densities = data.densities
total_energies = data.total_energies
grids = data.grids

def nn_functions(*layer_gens):
    """Takes tuples of (init_function, predict_function)
    and returns the init function and list of predictions"""
    num_layers = len(layer_gens)
    init_funcs = [layer[0] for layer in layer_gens]
    predict_funcs = [layer[1] for layer in layer_gens]

    def init_all_params(key):
        keys = random.split(key, num_layers)
        return [init(k) for init, k in zip(init_funcs, keys)]

    return init_all_params, predict_funcs

init_params, func_list = nn_functions(
    get_global_conv_layer(16, min_xi=0.1,max_xi=2.385345,dx=0.08,grids=grids),
    get_conv_layer(3, 16, 16, silu),
    get_conv_layer(3, 16, 128, silu),
    get_conv_layer(3, 128, 128, silu),
    get_maxpool2_flatten_dense(513//2*128, 128, silu),
    get_dense_layer(128, 1, lambda i: i)
)

@jit
def predict(params, x):
    activations = x
    for func, param in zip(func_list, params):
        activations = func(param, activations)
    return activations.reshape(-1)

@jit
def MSE_Loss(params, x, target):
    """Mean-squared error"""
    return jnp.mean((predict(params, x) - target)**2)

@jit
def grad_descent_update(params, batch_x, batch_y, eta):
    """Perform an iteration of gradient descent, with learning rate eta."""
    
    grads = grad(MSE_Loss)(params, batch_x, batch_y)
    return [w - eta*dw if not type(w) is tuple 
            else (w[0]-eta*dw[0], w[1]-eta*dw[1]) #Handle dense layer case, two parameter arrays
                for w, dw in zip(params, grads)]

train_distances = [128, 384]
train_mask = jnp.isin(data.distances_x100, train_distances) #Only train at 1.28 and 3.84
validation_distances = [296]
validation_mask = jnp.isin(data.distances_x100, validation_distances) #Only train at 1.28 and 3.84

#Hyperparameters: seed=1 and eta=0.1 are optimal from cross-validation
seed = 1 #parameter seed
eta = 0.1 #training rate
epochs = 1000 #number of training epochs

#Cross_validate
#seeds = range(25)
#etas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

#best_stats = {}
#best_valid_error = jnp.inf
#for eta in etas:
#    for seed in seeds:
init_key = random.PRNGKey(seed)
params = init_params(init_key)
for i in range(epochs):
    params = grad_descent_update(params, densities[train_mask,:], total_energies[train_mask], eta)
    if i%10==0:
        print('Training Cost:', MSE_Loss(params, densities[train_mask,:], total_energies[train_mask]))
print('Training Cost:', MSE_Loss(params, densities[train_mask,:], total_energies[train_mask]))
print('Validation Cost:', MSE_Loss(params, densities[validation_mask,:], total_energies[validation_mask]))
#train_cost = MSE_Loss(params, densities[train_mask,:], total_energies[train_mask])
#valid_cost = MSE_Loss(params, densities[validation_mask,:], total_energies[validation_mask])
#if valid_cost < best_valid_error:
#    best_valid_error = valid_cost
#    best_stats = {'seed':seed, 'eta':eta, 'train_cost':train_cost, 'valid_cost':valid_cost}

#print("------------------------")
#print("Best hyperparameters:")
#for k, v in best_stats.items():
#    print(k,':',v)

result_energies = predict(params, densities)

#Nuclear-nuclear repulsion energy, from the 1D exponential interaction
nuclear_energy = utils.get_nuclear_interaction_energy_batch(
    data.locations,
    data.nuclear_charges,
    interaction_fn=utils.exponential_coulomb)


plt.plot(data.distances, nuclear_energy+total_energies, linestyle='dashed', color='black', label='Exact')
plt.plot(data.distances, nuclear_energy+result_energies, color='purple', label='Convolutional network trained on D=1.28 and 3.84')
plt.plot(data.distances[train_mask], (nuclear_energy+total_energies)[train_mask], marker='D', linestyle='None')
plt.plot(data.distances[validation_mask], (nuclear_energy+total_energies)[validation_mask], marker='^', linestyle='None')

plt.xlabel('Distance')
plt.ylabel('$E + E_{nn}$')
plt.legend()

plt.show()