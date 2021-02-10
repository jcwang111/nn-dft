"""A reimplementation, also using JAX, of Fig S3(a) in Li et al., “Kohn-Sham equations as regularizer:
        Building prior knowledge into machine-learned physics,” arXiv:2009.08551 (2020),"""

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.nn import sigmoid, relu, softplus, silu, elu
from jax.scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from jax_dft import datasets
from jax_dft import utils
from jax_dft import np_utils
import scipy

from pureML_layers import *

#Retrieve data
data = datasets.Dataset(path='data/h2', num_grids=513)
densities = data.densities
total_energies = data.total_energies
grids = data.grids

def nn_functions(*layer_gens):
    """Takes tuples of (init_function, predict_function)
    and puts them together"""
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

train_densities = densities[train_mask,:]
train_energies = total_energies[train_mask]
valid_densities = densities[validation_mask,:]
valid_energies = total_energies[validation_mask]

#Hyperparameters
#init_key = random.PRNGKey(1) #parameter seed
#eta = 0.1 #training rate
epochs = 20 #number of training epochs

#Cross_validate
seeds = random.split(random.PRNGKey(68000), 50)
#etas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

best_stats = {}
best_valid_error = [np.inf]
best_valid_params = {'params':None, 'seed':None, 'valid_cost':None, 'calls': 0}
#for eta in etas:
#for init_key in seeds:
init_key = np.array([3483222781, 1411702757], dtype=np.uint32)
params = init_params(init_key)
spec, flatten_init_params = np_utils.flatten(params) 
grad_MSE = grad(MSE_Loss)

calls = [0]
def flatten_fn_grad(flatten_params):
    unflat_params = np_utils.unflatten(spec, flatten_params)
    valid_loss = MSE_Loss(unflat_params, valid_densities, valid_energies)
    calls[0] += 1
    if valid_loss < best_valid_error[0]:
        best_valid_error[0] = valid_loss
        best_valid_params['params'] = unflat_params
        best_valid_params['seed'] = init_key
        best_valid_params['valid_cost'] = valid_loss
        best_valid_params['calls'] = calls[0]

    return MSE_Loss(unflat_params, train_densities, train_energies), np_utils.flatten(grad_MSE(unflat_params, train_densities, train_energies))[1]

final_params, train_cost, info = scipy.optimize.fmin_l_bfgs_b(
    flatten_fn_grad,
    x0=np.array(flatten_init_params),
    # Maximum number of function evaluations.
    maxfun=epochs,
    factr=1,
    m=20,
    pgtol=1e-14)

params = np_utils.unflatten(spec, final_params)

print('Training Cost:', MSE_Loss(params, densities[train_mask,:], total_energies[train_mask]))
print('Validation Cost:', MSE_Loss(params, densities[validation_mask,:], total_energies[validation_mask]))
'''valid_cost = MSE_Loss(params, densities[validation_mask,:], total_energies[validation_mask])
    if valid_cost < best_valid_error:
        best_valid_error = valid_cost
        best_stats = {'seed':init_key, 'info':info, 'train_cost':train_cost, 'valid_cost':valid_cost}'''

#print("------------------------")
#print("Best hyperparameters:")
#for k, v in best_stats.items():
#    print(k,':',v)

print(best_valid_params)
params = best_valid_params['params']
print('Training Cost:', MSE_Loss(best_valid_params['params'], densities[train_mask,:], total_energies[train_mask]))
print('Validation Cost:', MSE_Loss(best_valid_params['params'], densities[validation_mask,:], total_energies[validation_mask]))
result_energies = predict(params, densities)

#Nuclear-nuclear repulsion energy, from the 1D exponential interaction
nuclear_energy = utils.get_nuclear_interaction_energy_batch(
    data.locations,
    data.nuclear_charges,
    interaction_fn=utils.exponential_coulomb)


plt.plot(data.distances, nuclear_energy+total_energies, linestyle='dashed', color='black', label='Exact')
plt.plot(data.distances, nuclear_energy+result_energies, color='purple', label='Training point with lowest validation cost')
plt.plot(data.distances[train_mask], (nuclear_energy+total_energies)[train_mask], marker='D', linestyle='None')
plt.plot(data.distances[validation_mask], (nuclear_energy+total_energies)[validation_mask], marker='^', linestyle='None')

plt.xlabel('Distance')
plt.ylabel('$E + E_{nn}$')
#plt.legend()

plt.show()