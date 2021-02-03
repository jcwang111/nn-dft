'''A simple one-layer network implementation for the H2 data.'''

import numpy as np
import matplotlib.pyplot as plt
from jax_dft import datasets
from jax_dft import utils

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def SiLU(z):
    return z / (1 + np.exp(-z))

def SiLU_prime(z):
    return (1+np.exp(-z)+z*np.exp(-z)) / (1+np.exp(-z))**2

def ReLU(z):
    return z * (z > 0)

def ReLU_prime(z):
    return 1 * (z > 0)

def Softplus(z):
    return np.log(1+np.exp(z))

def Softplus_prime(z):
    return 1/(1+np.exp(-z))

def ELU(z, alpha):
    return np.piecewise(z,[z>0, z<=0],[lambda x:x, lambda x:alpha*(np.exp(x)-1)])

def ELU_prime(z, alpha):
    return np.piecewise(z,[z>0, z<=0],[1, lambda x:alpha*np.exp(x)])

def L2_cost(result, y):
    #result and y are column vectors
    return 1/(2*y.size) * (y-result) @ (y-result).T

class OneLayerNN():
    def __init__(self, sizes, activation, activation_prime):
        '''Args:
            sizes: numpy array of shape (3,), sizes of layers
            activation: activation function to use, e.g. sigmoid
            activation_prime: derivative of activation function
        '''
        #sizes = (513, 8, 1) means 8-neuron hidden network
        self.sizes = sizes
        self.num_weights = len(sizes) - 1 #len(weights) and len(b)
        self.weights = [np.random.normal(0, 0.1, (sizes[i],sizes[i-1])) for i in range(1,len(sizes))]
        self.b = [np.random.normal(0, 0.1, (sizes[i],1)) for i in range(1,len(sizes))]
        self.activation = activation
        self.activation_prime = activation_prime
        #print([w.shape for w in self.weights])
        #print([b.shape for b in self.b])


    def __call__(self, x):
        '''Return the result of the network on an input,
           where the first dimension is the batch and the
           second is the features'''
        a = x.T

        for i in range(self.num_weights - 1):
            a = self.activation(self.weights[i]@a + self.b[i])

        return self.weights[-1]@a + self.b[-1]

    def backprop(self, x, y):
        #Forward pass
        activations = [x.T]
        z_s = [] #pre-activation function input

        for i in range(self.num_weights):
            z_s.append(self.weights[i]@activations[i] + self.b[i])
            activations.append(self.activation(z_s[i]))

        #Backpropagate
        delta = [(z_s[-1]-y)] #Assume no activation on the final output
        for i in range(self.num_weights-1, 0, -1):
            delta.append((self.weights[i].T @ delta[-1]) * self.activation_prime(z_s[i-1]))

        delta.reverse()
        activations[-1] = z_s[-1]
        return activations, delta

    def gd(self, batch_x, batch_y, eta):
        assert batch_x.shape[0]==batch_y.shape[0]
        n = batch_y.size
        activations, delta = self.backprop(batch_x, batch_y)
        #print([a.shape for a in activations])
        #print([d.shape for d in delta])

        for i in range(n):
            for l in range(self.num_weights-1, -1, -1):
                self.weights[l] = self.weights[l] - eta/n* delta[l][:,[i]]@activations[l][:,[i]].T
                self.b[l] = self.b[l] - eta/n* delta[l][:,[i]]

    def train(self, batch_x, batch_y, eta, iterations, print_cost=False):
        for i in range(iterations):
            self.gd(batch_x, batch_y, eta)
            result = self.__call__(batch_x)
            if print_cost and i%10==0: print('Training Cost:', L2_cost(result, batch_y)[0][0])
        return result

data = datasets.Dataset(path='data/h2/', num_grids=513)
densities = data.densities

nn1 = OneLayerNN((513, 12, 12, 1), SiLU, SiLU_prime)
nn1.train(densities, data.total_energies, 0.4, 1000, True)
nn1_energies = nn1(densities).reshape(-1)

'''train_distances = [128, 384]
nn2 = OneLayerNN((513, 8, 1), sigmoid, sigmoid_prime)
mask = np.isin(data.distances_x100, train_distances) #Only train at 1.28 and 3.84
nn2.train(densities[mask,:], data.total_energies[mask], 0.3, 300, True)
nn2_energies = nn2(densities).reshape(-1)'''

#To get nuclear_energy, which is added to total_energies
nuclear_energy = utils.get_nuclear_interaction_energy_batch(
    data.locations,
    data.nuclear_charges,
    interaction_fn=utils.exponential_coulomb)


plt.plot(data.distances, nuclear_energy+data.total_energies, linestyle='dashed', color='black', label='Exact')
plt.plot(data.distances, nuclear_energy+nn1_energies, color='green', label='Neural network trained on all 72 sample densities')
#plt.plot(data.distances, nuclear_energy+nn2_energies, color='purple', label='Neural network trained on d=1.28 and 3.84')
plt.xlabel('Distance')
plt.ylabel('Energy')
plt.legend()
#plt.savefig("1layerNN.png")
plt.show()