import jax 
import numpy as np
import os
from .config import cfg
from .dynamics import *
from tqdm.auto import tqdm
# from functools import partial
# import jax.experimental.host_callback
eps = 1E-6

class Simulation:
    def __init__(self,
                 samples=None,
                 model=None, 
                 Tf=None,
                 dT=None,
                 Ntherm=None,
                 Nsamples=None,
                 Nsweeps=None,
                 resultsfolder=None):
        self.samples = samples
        self.model = model
        self.Tf = Tf
        self.dT = dT
        self.Ntherm = Ntherm
        self.Nsamples = Nsamples
        self.Nsweeps = Nsweeps
        self.resultsfolder=resultsfolder
        os.makedirs(resultsfolder,exist_ok=True)

    def _tree_flatten(self):
        children = (self.samples,
                    self.model,
                    self.Tf,
                    self.dT,
                    self.Ntherm,
                    self.Nsamples,
                    self.Nsweeps)  # arrays / dynamic values
        aux_data = {'resultsfolder':self.resultsfolder}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
jax.tree_util.register_pytree_node(Simulation,
                               Simulation._tree_flatten,
                               Simulation._tree_unflatten)

def save_hyperparameters(sim):
    filename = sim.resultsfolder+f'hyperp.txt'
    os.system(f'rm -f {filename}')
    with open(filename,'a') as f:
        np.savetxt(f,[
                      sim.model.h,
                      cfg.mc.T0,
                      cfg.mc.Tf,
                      cfg.mc.dT,
                      cfg.mc.Ntherm0,
                      cfg.mc.Ntherm,
                      cfg.mc.Nsamples,
                      cfg.mc.Nsweeps,
                      ],delimiter='\t',fmt='%.3f')
    return


@jax.jit
def do_Nsweeps(idx,sim):
    sim.model = jax.lax.fori_loop(lower=0,
                        upper=sim.Nsweeps,
                        body_fun=sweep,
                        init_val=sim.model)
    return sim

@jax.jit
def thermalisation(sim):
    sim =jax.lax.fori_loop(lower=0,
                        upper=sim.Ntherm,
                        body_fun=do_Nsweeps,
                        init_val=sim)
    return sim

@jax.jit
def _run_sim(idx,sim):
    sim.model = jax.lax.fori_loop(lower=0,
                        upper=sim.Nsweeps,
                        body_fun=sweep,
                        init_val=sim.model)
    sim.samples = sim.samples.at[idx].set(sim.model.spins)
    return sim

def run_sim(sim):
    sim = jax.lax.fori_loop(lower=0,
                            upper=sim.Nsamples,
                            body_fun=_run_sim,
                            init_val=sim)
    fname = f'{sim.resultsfolder}T{sim.model.T:.2f}'
    jnp.save(file=fname,arr=sim.samples)
    return sim

def annealing(sim):
    sim.Ntherm = cfg.mc.Ntherm
    while(sim.model.T > sim.Tf - eps):
        print(f'{sim.model.T=:.2f}')
        sim = thermalisation(sim)
        sim = run_sim(sim)
        sim.model.T -= sim.dT
    return sim

