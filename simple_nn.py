'''A simple one-layer network implementation for the H2 data.'''

import numpy as np
import matplotlib.pyplot as plt
from jax_dft import datasets
from jax_dft import utils

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def L2_cost(result, y):
    #result and y are column vectors
    return 1/(2*y.size) * (y-result) @ (y-result).T

class OneLayerNN():
    def __init__(self, sizes):
        #sizes = (513, 100, 1) means 100-neuron hidden network
        self.weights = [np.random.normal(0, 0.1, (sizes[i],sizes[i-1])) for i in range(1,len(sizes))]
        self.b = [np.random.normal(0, 0.1, (sizes[i])) for i in range(1,len(sizes))]
        #print([w.shape for w in self.weights])
        #print([b.shape for b in self.b])


    def __call__(self, x):
        '''Return the result of the network on an input'''
        a1 = sigmoid(self.weights[0]@x + self.b[0].reshape(-1,1))
        return self.weights[1]@a1 + self.b[1].reshape(-1,1)

    def backprop(self, x, y):
        #Forward pass
        activations = [x]
        #Layer 1
        z0 = self.weights[0]@x + self.b[0].reshape(-1,1)
        activations.append(sigmoid(z0))
        #Result
        activations.append(self.weights[1]@activations[1] + self.b[1].reshape(-1,1))

        #Backpropagate
        delta = []
        #Result
        delta.append((activations[-1]-y)) #grad_cost = a_final - y
        #Layer 1
        delta.append((self.weights[-1].T @ delta[-1]) * sigmoid_prime(z0))
        delta.reverse()

        return activations, delta

    def gd(self, batch_x, batch_y, eta):
        assert batch_x.shape[1]==batch_y.shape[0]
        n = batch_y.size
        activations, delta = self.backprop(batch_x, batch_y)
        #print([a.shape for a in activations])
        #print([d.shape for d in delta])

        for i in range(n):
            self.weights[1] = self.weights[1] - eta/n* delta[1][:,[i]]@activations[1][:,[i]].T
            self.b[1] = self.b[1] - eta/n* delta[1][:,i]

            self.weights[0] = self.weights[0] - eta/n* delta[0][:,[i]]@activations[0][:,[i]].T
            self.b[0] = self.b[0] - eta/n* delta[0][:,i]

    def train(self, batch_x, batch_y, eta, iterations, print_cost=False):
        for i in range(iterations):
            self.gd(batch_x, batch_y, eta)
            result = self.__call__(batch_x)
            if print_cost and i%10==0: print('Cost:', L2_cost(x, y)[0][0])
        return result

data = datasets.Dataset(path='data/h2/', num_grids=513)
densities = data.densities.T #My implementation uses the batches as columns

nn1 = OneLayerNN((513, 8, 1))
nn1.train(densities, data.total_energies, 0.3, 300)
nn1_energies = nn1(densities).reshape(-1)

train_distances = [128, 384]
nn2 = OneLayerNN((513, 8, 1))
mask = np.isin(data.distances_x100, train_distances) #Only train at 1.28 and 3.84
nn2.train(densities[:,mask], data.total_energies[mask], 0.3, 300)
nn2_energies = nn2(densities).reshape(-1)

#To get nuclear_energy, which is added to total_energies
nuclear_energy = utils.get_nuclear_interaction_energy_batch(
    data.locations,
    data.nuclear_charges,
    interaction_fn=utils.exponential_coulomb)


plt.plot(data.distances, nuclear_energy+data.total_energies, linestyle='dashed', color='black', label='Exact')
plt.plot(data.distances, nuclear_energy+nn1_energies, color='green', label='Neural network trained on all 72 sample densities')
plt.plot(data.distances, nuclear_energy+nn2_energies, color='purple', label='Neural network trained on d=1.28 and 3.84')
plt.xlabel('Distance')
plt.ylabel('Energy')
plt.legend()
#plt.savefig("1layerNN.png")
plt.show()