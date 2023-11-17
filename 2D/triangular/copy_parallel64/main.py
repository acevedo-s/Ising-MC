import jax
import numpy as np
from mc import *
import sys
import time

eps = 1E-6
resultsfolder0 = f'/scratch/sacevedo/Ising-triang/canonical/L{cfg.model.L}_seed_{cfg.seed}/'  

key0 = jax.random.PRNGKey(cfg.seed)
spins0 = jax.numpy.zeros(shape=(cfg.model.L,
                                cfg.model.L),
                                dtype=int)
samples0 = jax.numpy.zeros(shape=(cfg.mc.Nsamples,
                                  cfg.model.L,
                                  cfg.model.L),
                          dtype=int)

key0, subkey = jax.random.split(key0, num=2)

model = Model(key=key0,
              spins=spins0,
              T=cfg.mc.T0,
              L=cfg.model.L,
              h=cfg.model.h,
              )
model = init_state(model)
# print(model.E/model.L/model.L)

sim = Simulation(samples=samples0,
                 model=model,
                 Tf=cfg.mc.Tf,
                 dT=cfg.mc.dT,
                 Ntherm=cfg.mc.Ntherm0,
                 Nsamples=cfg.mc.Nsamples,
                 Nsweeps=cfg.mc.Nsweeps,
                 resultsfolder=resultsfolder0
                 )

if 1:
  save_hyperparameters(sim)
  print(f'thermalisation started ; {cfg.seed=}')
  start = time.time()
  sim = thermalisation(sim)
  print(f'thermalisation took {(time.time()-start)/60:.2f} minutes')

if 1:
  start = time.time()
  sim = annealing(sim)
  print(f'annealing took {(time.time()-start)/60:.2f} minutes')

sim.model = energy(sim.model)
print('E_final=',sim.model.E/model.L/model.L)
print(sim.model.spins)
# if cfg.model.b == 0 and cfg.model.h==0:
#   E = (model.L* f_exact_PBC(1/model.T,model.L)+
#        model.T*model.L*s_exact_PBC(1/model.T,model.L))
#   print(f'E_exact = {E:.5f}')
#   states = sim.samples
#   Es = energy_chain_PBC(states,normalized=0)
#   meanE = np.mean(Es)
#   stdE = np.std(Es)
#   print(f'<E>={meanE:.3f}+-{stdE:.3f}')




