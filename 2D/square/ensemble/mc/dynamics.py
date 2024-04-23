import jax
import jax.numpy as jnp
from jax.experimental.host_callback import call
from .model import *

eps = 1E-8

@jax.jit
def update_state(rand_i,rand_j,model):
  model.spins = model.spins.at[rand_i,rand_j].set(
    (-1) * model.spins[rand_i,rand_j]
    )
  # model.E += model.dE
  return model

@jax.jit
def do_nothing(rand_i,rand_j,model):
  return model

@jax.jit
def T_flip_spin(rand_i,rand_j,model):
  model.key, subkey = jax.random.split(model.key, num=2)
  r = jax.random.uniform(subkey)
  # call(lambda x: print(f'{x=}'),r)
  P = jnp.exp((-1/model.T) * model.dE)
  model = jax.lax.cond(r<P,
                       update_state,
                       do_nothing,
                       rand_i,rand_j,model)
  return model

@jax.jit
def delta_E(rand_i,rand_j,model):
  ### triangular
  # model.dE =  -(2 * model.spins[rand_i,rand_j] * 
              #  ( model.spins[(rand_i+1)%model.L,rand_j] 
              #   +model.spins[(rand_i+1)%model.L,(rand_j-1)%model.L]
              #   +model.spins[rand_i,(rand_j-1)%model.L]
              #   +model.spins[(rand_i-1)%model.L,rand_j]
              #   +model.spins[(rand_i-1)%model.L,(rand_j+1)%model.L]
              #   +model.spins[rand_i,(rand_j+1)%model.L]
              #   )
              #   )
  ### square (ferro)
  model.dE =  (2 * model.spins[rand_i,rand_j] * 
              (model.spins[(rand_i+1)%model.L,rand_j] +
               model.spins[rand_i,(rand_j+1)%model.L])
               )
  model.dE += (2 * model.spins[rand_i,rand_j] * 
              (model.spins[(rand_i-1)%model.L,rand_j] +
               model.spins[rand_i,(rand_j-1)%model.L])
               )
  # call(lambda x: print(f'dE={x}'),model.dE)
  # dh = (model.h*2*model.spins[rand_idx])
  # model.dE += dh
  # model.dE = model.dE[0]
  return model

@jax.jit
def single_spin_flip(idx,model):
  model.key,subkey = jax.random.split(model.key, num=2)
  rand_i,rand_j = jax.random.randint(subkey,(2,), minval=0,maxval=model.L)
  # call(lambda x: print(f'(i,j)={(x[0],x[1])}'),(rand_i,rand_j))
  model = delta_E(rand_i,rand_j,model)
  # call(lambda x: print(f'dE={x}'),model.dE)
  model = jax.lax.cond(model.dE<=eps,
               update_state,
               T_flip_spin,
               rand_i,rand_j,model)
  return model

@jax.jit
def sweep(idx,model):
  model = jax.lax.fori_loop(lower = 0,
                    upper = model.L,
                    body_fun = single_spin_flip,
                    init_val = model)
  return model



