"""A reimplementation, also using JAX, of Fig S3(a) in Li et al., “Kohn-Sham equations as regularizer:
        Building prior knowledge into machine-learned physics,” arXiv:2009.08551 (2020),"""

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.nn import sigmoid, relu, softplus, silu, elu
from jax import scipy
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
            else (w[0]-eta*dw[0], w[1]-eta*dw[1]) #Handle dense layer case, two arrays w and b
                for w, dw in zip(params, grads)]

train_distances = [128, 384]
train_mask = jnp.isin(data.distances_x100, train_distances) #Only train at 1.28 and 3.84
validation_distances = [296]
validation_mask = jnp.isin(data.distances_x100, validation_distances) #Only train at 1.28 and 3.84

train_densities = densities[train_mask,:]
train_energies = total_energies[train_mask]
valid_densities = densities[validation_mask,:]
valid_energies = total_energies[validation_mask]

def train_and_return(eta, init_key):
    """ Initializes a new model, trains it, and returns results.
    Args:
        eta: training rate
        init_key: jnp array (2,), a PRNGKey()
    Returns:
        dict containing:
            params: list of trained parameter arrays
            train_loss: float, final training cost
            valid_loss: float, final validation cost
            eta: eta argument
            seed: init_key argument
            loss_record: (parameters, step number) for every 10th BFGS step
            num_steps: number of BFGS iterations taken
    """

    params = init_params(init_key)
    spec, flat_init_params = np_utils.flatten(params) 
    grad_MSE = grad(MSE_Loss)

    loss_record = []
    def loss_grad_fn(flat_params):
        unflat_params = np_utils.unflatten(spec, flat_params)

        loss_grad_fn.step += 1
        #if loss_grad_fn.step % 10 == 0: #Save checkpoint every 10
            #valid_loss = MSE_Loss(unflat_params, valid_densities, valid_energies)
            #loss_record.append((unflat_params, loss_grad_fn.step))

        return MSE_Loss(unflat_params, train_densities, train_energies), np_utils.flatten(grad_MSE(unflat_params, train_densities, train_energies))[1]
            
    loss_grad_fn.step = 0

    print(flat_init_params.shape)
    epochs = 20
    final_params, train_cost, info = scipy.optimize.fmin_l_bfgs_b(
                loss_grad_fn,
                x0=np.array(flat_init_params),
                # Maximum number of function evaluations.
                maxfun=epochs,
                factr=1,
                m=20,
                pgtol=1e-14)
    
    params = np_utils.unflatten(spec, result.x)
    valid_loss = MSE_Loss(params, valid_densities, valid_energies)

    print('Training Cost:', train_cost)
    print('Validation Cost:', valid_loss)
    return {'params':params, 'train_loss':train_cost, 'valid_loss':valid_loss, 'eta':eta, 'seed':init_key,
            'loss_record':loss_record, 'num_steps':loss_grad_fn.step}

#Cross validation
seeds = random.split(random.PRNGKey(68000), 30)
etas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

best_valid_loss = jnp.inf
best_result = None
for eta in etas:
    for init_key in seeds:
        result = train_and_return(eta, init_key)
        if result['valid_loss'] < best_valid_loss:
            best_result = result

print("------------------------")
print("Best hyperparameters:")
for k, v in best_stats.items():
    if k != 'params' and k != 'loss_record':
        print(k,':',v)


params = best_valid_params['params']
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
plt.legend()

plt.show()