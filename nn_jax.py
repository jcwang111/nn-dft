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
data = datasets.Dataset(path='data/H3/')
densities = data.densities
total_energies = data.total_energies

train_distances = [200, 344]
train_mask = jnp.isin(data.distances_x100, train_distances) #Only train at 1.28 and 3.84
validation_distances = [296]
validation_mask = jnp.isin(data.distances_x100, validation_distances) #Only train at 1.28 and 3.84

assert(densities[train_mask].shape[0] == len(train_distances))

#Hyperparameters
orig_init_key = random.PRNGKey(34000) #parameter seed
layers = (513, 10, 10, 10, 1) #network structure
#eta = 0.5 #training rate
act_func = silu #activation function
epochs = 1000 #number of training epochs

#Cross_validate
seeds = random.split(orig_init_key, 50)
etas = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]

best_stats = {}
best_params = None
best_valid_error = jnp.inf
for eta in etas:
    for init_key in seeds:
        params = init_network_params(layers, init_key)
        for i in range(epochs):
            params = grad_descent_update(params, densities[train_mask,:], total_energies[train_mask], eta)
            #if i%200==0:
                #print('Training Cost:', L2_cost(params, densities[train_mask,:], total_energies[train_mask]))
        #print('Training Cost:', L2_cost(params, densities[train_mask,:], total_energies[train_mask]))
        #print('Validation Cost:', L2_cost(params, densities[validation_mask,:], total_energies[validation_mask]))
        train_cost = L2_cost(params, densities[train_mask,:], total_energies[train_mask])
        valid_cost = L2_cost(params, densities[validation_mask,:], total_energies[validation_mask])
        if valid_cost < best_valid_error:
            best_valid_error = valid_cost
            best_stats = {'seed':init_key, 'eta':eta, 'train_cost':train_cost, 'valid_cost':valid_cost}
            best_params = params

print("------------------------")
print("Best hyperparameters:")
for k, v in best_stats.items():
    print(k,':',v)
print(train_distances)

result_energies = batch_predict(best_params, densities)

#Nuclear-nuclear repulsion energy, from the 1D exponential interaction
nuclear_energy = utils.get_nuclear_interaction_energy_batch(
    data.locations,
    data.nuclear_charges,
    interaction_fn=utils.exponential_coulomb)


plt.plot(data.distances, nuclear_energy+total_energies, linestyle='dashed', color='black', label='Exact')
plt.plot(data.distances, nuclear_energy+result_energies, color='green',label='Network trained on total energy')
plt.plot(data.distances[train_mask], (nuclear_energy+total_energies)[train_mask], marker='D', linestyle='None')
plt.plot(data.distances[validation_mask], (nuclear_energy+total_energies)[validation_mask], marker='^', linestyle='None')
plt.plot()
plt.xlabel('Distance')
plt.ylabel('$E + E_{nn}$')
plt.legend()

#plt.savefig('h3_5pts.png')
plt.show()