import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0'

import glob
import pickle
import time
import jax
from jax import random
from jax import tree_util
from jax.config import config
import jax.numpy as jnp
from jax_dft import datasets
from jax_dft import jit_scf
from jax_dft import losses
from jax_dft import neural_xc
from jax_dft import np_utils
from jax_dft import scf
from jax_dft import utils
from jax_dft import xc
import matplotlib.pyplot as plt
import numpy as np
import scipy



# Set the default dtype as float64
#config.update("jax_debug_nans", True)
#config.update('jax_enable_x64', True)

#train_distances = [128, 384]  #@param
train_distances = [128, 384]

dataset = datasets.Dataset(path='h2/', num_grids=513)
grids = dataset.grids
train_set = dataset.get_molecules(train_distances)

print(train_set.total_energy.shape, train_set.density.shape, train_set.nuclear_charges.shape)
#@title Initial density
initial_density = scf.get_initial_density(train_set, method='noninteracting')
print(initial_density.shape)



#@title Initialize network
network = neural_xc.build_global_local_conv_net(
#network = build_global_Biased_local_conv_net(
    num_global_filters=16,
    num_local_filters=16,
    num_local_conv_layers=2,
    activation='swish',
    grids=grids,
    minval=0.1,
    maxval=2.385345,
    downsample_factor=0,
    apply_negativity_transform=True)
network = neural_xc.wrap_network_with_self_interaction_layer(
    network, grids=grids, interaction_fn=utils.exponential_coulomb)
init_fn, neural_xc_energy_density_fn = neural_xc.global_functional(
    network, grids=grids)
init_params = init_fn(random.PRNGKey(0))
initial_checkpoint_index = 0
spec, flatten_init_params = np_utils.flatten(init_params)
print(f'number of parameters: {len(flatten_init_params)}')



#@markdown The number of Kohn-Sham iterations in training.
num_iterations = 15 #@param{'type': 'integer'}
#@markdown The density linear mixing factor.
alpha = 0.5 #@param{'type': 'number'}
#@markdown Decay factor of density linear mixing factor.
alpha_decay = 0.9 #@param{'type': 'number'}
#@markdown The number of density differences in the previous iterations to mix the
#@markdown density. Linear mixing is num_mixing_iterations = 1.
num_mixing_iterations = 1 #@param{'type': 'integer'}
#@markdown The stopping criteria of Kohn-Sham iteration on density.
density_mse_converge_tolerance = -1. #@param{'type': 'number'}
#@markdown Apply stop gradient on the output state of this step and all steps
#@markdown before. The first KS step is indexed as 0. Default -1, no stop gradient
#@markdown is applied.
stop_gradient_step=-1 #@param{'type': 'integer'}

def _kohn_sham(flatten_params, locations, nuclear_charges, initial_density):
  return jit_scf.kohn_sham(
      locations=locations,
      nuclear_charges=nuclear_charges,
      num_electrons=dataset.num_electrons,
      num_iterations=num_iterations,
      grids=grids,
      xc_energy_density_fn=tree_util.Partial(
          neural_xc_energy_density_fn,
          params=np_utils.unflatten(spec, flatten_params)),
      interaction_fn=utils.exponential_coulomb,
      # The initial density of KS self-consistent calculations.
      initial_density=initial_density,
      alpha=alpha,
      alpha_decay=alpha_decay,
      enforce_reflection_symmetry=True,
      num_mixing_iterations=num_mixing_iterations,
      density_mse_converge_tolerance=density_mse_converge_tolerance,
      stop_gradient_step=stop_gradient_step)
_batch_jit_kohn_sham = jax.vmap(_kohn_sham, in_axes=(None, 0, 0, 0))

grids_integration_factor = utils.get_dx(grids) * len(grids)

@jax.jit
def loss_fn(
    flatten_params, locations, nuclear_charges,
    initial_density, target_energy, target_density):
  """Get losses."""
  states = _batch_jit_kohn_sham(
      flatten_params, locations, nuclear_charges, initial_density)
  # Energy loss
  loss_value = losses.trajectory_mse(
      target=target_energy,
      predict=states.total_energy[
          # The starting states have larger errors. Ignore the number of 
          # starting states (here 10) in loss.
          :, 10:],
      # The discount factor in the trajectory loss.
      discount=0.9) / dataset.num_electrons
  # Density loss
  loss_value += losses.mean_square_error(
      target=target_density, predict=states.density[:, -1, :]
      ) * grids_integration_factor / dataset.num_electrons
  return loss_value

value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

#@markdown The frequency of saving checkpoints.
save_every_n = 5 #@param{'type': 'integer'}

loss_record = []

##################################################
valid_distances = [296]
valid_set = dataset.get_molecules(valid_distances)
valid_initial_density = scf.get_initial_density(valid_set, method='noninteracting')

def validation_loss(flatten_params):
  return loss_fn(flatten_params,
      locations=valid_set.locations,
      nuclear_charges=valid_set.nuclear_charges,
      initial_density=valid_initial_density,
      target_energy=valid_set.total_energy,
      target_density=valid_set.density)
  
best_values = {'valid_cost':jnp.inf}
###################################################  

def np_value_and_grad_fn(flatten_params):
  """Gets loss value and gradient of parameters as float and numpy array."""
  start_time = time.time()
  # Automatic differentiation.
  train_set_loss, train_set_gradient = value_and_grad_fn(
      flatten_params,
      locations=train_set.locations,
      nuclear_charges=train_set.nuclear_charges,
      initial_density=initial_density,
      target_energy=train_set.total_energy,
      target_density=train_set.density)
  step_time = time.time() - start_time
  step = initial_checkpoint_index + len(loss_record)
  print(f'step {step}, loss {train_set_loss} in {step_time} sec')

  # Save checkpoints.
  '''if len(loss_record) % save_every_n == 0:
    checkpoint_path = f'ckpt-{step:05d}'
    print(f'Save checkpoint {checkpoint_path}')
    with open(checkpoint_path, 'wb') as handle:
      pickle.dump(np_utils.unflatten(spec, flatten_params), handle)'''
  ################################
  if len(loss_record) % save_every_n == 0:
    valid_loss = validation_loss(flatten_params)
    print(valid_loss)
    if valid_loss < best_values['valid_cost']:
      best_values['valid_cost'] = valid_loss

      checkpoint_path = 'best_ckpt' #f'ckpt-{step:05d}'
      print(f'Save checkpoint {step:05d}')
      with open(checkpoint_path, 'wb') as handle:
        pickle.dump(np_utils.unflatten(spec, flatten_params), handle)
  ################################
  loss_record.append(train_set_loss)
  return train_set_loss, np.array(train_set_gradient)
  

#######################################
# Test grad
#######################################

grad_fn = jax.grad(loss_fn)

def train_grad_fn(flatten_params):
  return grad_fn(flatten_params,
      locations=train_set.locations,
      nuclear_charges=train_set.nuclear_charges,
      initial_density=initial_density,
      target_energy=train_set.total_energy,
      target_density=train_set.density)

init_params = init_fn(random.PRNGKey(0))
#spec, flatten_init_params = np_utils.flatten(init_params)
print(flatten_init_params)
print('G',train_grad_fn(flatten_init_params))
