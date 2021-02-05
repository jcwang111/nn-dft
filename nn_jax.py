import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.nn import sigmoid, relu, softplus, silu, elu
import matplotlib.pyplot as plt
from jax_dft import datasets
from jax_dft import utils

def random_layer_params(m, n, key, scale=1E-1):
    '''Generate weights and biases for a m->n layer'''
    w_key, b_key = random.split(key)
    return scale*random.normal(w_key, (n,m)), scale*random.normal(b_key, (n,))

def init_network_params(sizes, key):
    '''Generate all weights and biases from the list of layer sizes'''
    keys = random.split(key, len(sizes)-1)
    return [random_layer_params(m,n,k) for m,n,k in zip(sizes[:-1], sizes[1:], keys)]

def predict(params, x):
    """Returns a prediction of the neural network"""
    activation = x
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activation) + b
        activation = act_func(outputs)
    
    w_final, b_final = params[-1]
    return jnp.dot(w_final, activation) + b_final

@jit
def batch_predict(params, x):
    return vmap(predict, in_axes=(None, 0))(params, x).reshape(-1)

@jit
def L2_cost(params, x, y):
    return jnp.mean((batch_predict(params, x) - y)**2)

@jit
def grad_descent_update(params, batch_x, batch_y, eta):
    """Perform an iteration of gradient descent, with learning rate eta."""
    
    grads = grad(L2_cost)(params, batch_x, batch_y)
    return [(w - eta*dw, b - eta*db) for (w,b),(dw,db) in zip(params, grads)]

#Retrieve data
data = datasets.Dataset(path='data/h2/', num_grids=513)
densities = data.densities
total_energies = data.total_energies

train_distances = [128, 384]
train_mask = jnp.isin(data.distances_x100, train_distances) #Only train at 1.28 and 3.84
validation_distances = [296]
validation_mask = jnp.isin(data.distances_x100, validation_distances) #Only train at 1.28 and 3.84

#Hyperparameters
init_key = random.PRNGKey(5) #parameter seed
layers = (513, 10, 10, 10, 1) #network structure
eta = 0.5 #training rate
act_func = silu #activation function
epochs = 800 #number of training epochs

#Cross_validate
#seeds = [0, 1, 2, 3, 4, 5]
#etas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

#best_stats = {}
#best_valid_error = jnp.inf
#for eta in etas:
#    for seed in seeds:
#       init_key = random.PRNGKey(seed)
params = init_network_params(layers, init_key)
for i in range(epochs):
    params = grad_descent_update(params, densities[train_mask,:], total_energies[train_mask], eta)
    if i%200==0:
        print('Training Cost:', L2_cost(params, densities[train_mask,:], total_energies[train_mask]))
print('Training Cost:', L2_cost(params, densities[train_mask,:], total_energies[train_mask]))
print('Validation Cost:', L2_cost(params, densities[validation_mask,:], total_energies[validation_mask]))
#train_cost = L2_cost(params, densities[train_mask,:], total_energies[train_mask])
#valid_cost = L2_cost(params, densities[validation_mask,:], total_energies[validation_mask])
#if valid_cost < best_valid_error:
#    best_valid_error = valid_cost
#    best_stats = {'seed':seed, 'eta':eta, 'train_cost':train_cost, 'valid_cost':valid_cost}

#print("------------------------")
#print("Best hyperparameters:")
#for k, v in best_stats.items():
#    print(k,':',v)

result_energies = batch_predict(params, densities)

#Nuclear-nuclear repulsion energy, from the 1D exponential interaction
nuclear_energy = utils.get_nuclear_interaction_energy_batch(
    data.locations,
    data.nuclear_charges,
    interaction_fn=utils.exponential_coulomb)


plt.plot(data.distances, nuclear_energy+total_energies, linestyle='dashed', color='black', label='Exact')
plt.plot(data.distances, nuclear_energy+result_energies, color='green', label='Network trained on samples D=1.28 and D=3.84')
plt.plot(data.distances[train_mask], (nuclear_energy+total_energies)[train_mask], marker='D', linestyle='None')
plt.plot(data.distances[validation_mask], (nuclear_energy+total_energies)[validation_mask], marker='^', linestyle='None')
plt.plot()
plt.xlabel('Distance')
plt.ylabel('$E + E_{nn}$')
plt.legend()

plt.show()