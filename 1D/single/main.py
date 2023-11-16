import jax
import numpy as np
from mc import *
import sys
import time

eps = 1E-6
if np.isclose(cfg.model.J2,0):
  geometry = 'Ising-chain'
else:
  geometry = 'Ising-square'
resultsfolder0 = f'/scratch/sacevedo/{geometry}/canonical/L{cfg.model.L}_seed_{cfg.seed}/'
os.makedirs(resultsfolder0,exist_ok=True)

key0 = jax.random.PRNGKey(cfg.seed)
key0, subkey = jax.random.split(key0, num=2)

if cfg.mc.load_spins==0:
  spins0 = jax.numpy.zeros(shape=cfg.model.L,dtype=int)
elif cfg.mc.load_spins==1:
  spins0 = jnp.load(f'{resultsfolder0}T{cfg.mc.T0:.2f}.npy')[-1]
samples0 = jax.numpy.zeros(shape=(cfg.mc.Nsamples,cfg.model.L),dtype=int)

model = Model(key=key0,
              spins=spins0,
              T=cfg.mc.T0,
              h=cfg.model.h,
              L=cfg.model.L,
              J1=cfg.model.J1,
              J2=cfg.model.J2,)
if cfg.mc.load_spins==0:
  model = init_state(model)
elif cfg.mc.load_spins==1:
  model.T -= cfg.mc.dT
  # print(model.spins[:20])


sim = Simulation(samples=samples0,
                 model=model,
                 Tf=cfg.mc.Tf,
                 dT=cfg.mc.dT,
                 Ntherm=cfg.mc.Ntherm0,
                 Nsamples=cfg.mc.Nsamples,
                 Nsweeps=cfg.mc.Nsweeps,
                 resultsfolder=resultsfolder0
                 )

if cfg.mc.load_spins==0:
  save_hyperparameters(sim)
  print(f'thermalisation started ; {cfg.seed=}')
  start = time.time()
  sim = thermalisation(sim)
  print(f'thermalisation took {(time.time()-start)/60:.2f} minutes')

if True:
  start = time.time()
  sim = annealing(sim)
  print(f'annealing took {(time.time()-start)/60:.2f} minutes')
  # print(sim.samples[-1,:20])


# if cfg.model.b == 0 and cfg.model.h==0:
#   E = (model.L* f_exact_PBC(1/model.T,model.L)+
#        model.T*model.L*s_exact_PBC(1/model.T,model.L))
#   print(f'E_exact = {E:.5f}')
#   states = sim.samples
#   Es = energy_chain_PBC(states,normalized=0)
#   meanE = np.mean(Es)
#   stdE = np.std(Es)
#   print(f'<E>={meanE:.3f}+-{stdE:.3f}')




