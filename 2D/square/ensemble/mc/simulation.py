import jax 
import numpy as np
import os
from .dynamics import *
from tqdm.auto import tqdm
from params import *

# from functools import partial
# import jax.experimental.host_callback
eps = 1E-6

class Simulation:
    """Ntherm is initialized to Ntherm0, to equilibrate at high temperature,
    then updated to equilibrate between temperatures."""
    def __init__(self,
                 model=None, 
                 Tf=None,
                 dT=None,
                 Ntherm=None, 
                 resultsfolder=None,
                 ):
        self.model = model
        self.Tf = Tf
        self.dT = dT
        self.Ntherm = Ntherm
        self.resultsfolder=resultsfolder
        os.makedirs(resultsfolder,exist_ok=True)

    def _tree_flatten(self):
        children = (
                    self.model,
                    self.Tf,
                    self.dT,
                    self.Ntherm,
                    )  # arrays / dynamic values
        aux_data = {'resultsfolder':self.resultsfolder}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
jax.tree_util.register_pytree_node(Simulation,
                               Simulation._tree_flatten,
                               Simulation._tree_unflatten)

def save_hyperparameters(sim):
    filename = sim.resultsfolder+f'hyperp_L{sim.model.L}.txt'
    os.system(f'rm -f {filename}')
    with open(filename,'a') as f:
        np.savetxt(f,[
                      T0,
                      Tf,
                      dT,
                      Ntherm0,
                      Ntherm,
                      ],delimiter='\t',fmt='%.3f')
    return

@jax.jit
def thermalisation(sim):
  sim.model = jax.lax.fori_loop(lower=0,
                              upper=sim.Ntherm,
                              body_fun=sweep,
                              init_val=sim.model,
                              )
  return sim

def export_spins(sim):
  fname = f'{sim.resultsfolder}seed{sim.model.seed}_T{sim.model.T:.2f}'
  jnp.save(file=fname,arr=sim.model.spins)
  return

def annealing(sim):
  sim.Ntherm = Ntherm
  while(sim.model.T > sim.Tf - eps):
    if sim.model.square:
      if sim.model.T >= 2.5: # high T
        sim.dT = dT 
      elif sim.model.T <= 2.20 + dT/10/2: # low T
        sim.dT = dT  
      else: 
        sim.dT = dT/10 # critical zone
    print(f'T={sim.model.T:.2f}')
    sim = thermalisation(sim)
    export_spins(sim)
    sim.model.T -= sim.dT
  return sim

