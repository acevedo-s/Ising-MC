import jax
from mc import *
from params import * 

eps = 1E-6
resultsfolder0 = f'/scratch/sacevedo/{lattice}/canonical/L{L}/r_id{r_id}/seed{seed}/'  

key0 = jax.random.PRNGKey(seed)
spins0 = jax.numpy.zeros(shape=L,dtype=int)

key0, subkey = jax.random.split(key0, num=2)

model = Model(
              key=key0,
              spins=spins0,
              T=T0,
              L=L,
              )
model = init_state(model)
model = energy(model)
print(f'initial E:{model.E/model.L}')
sim = Simulation(
                 model=model,
                 Tf=Tf,
                 dT=dT,
                 Ntherm=Ntherm0,
                 resultsfolder=resultsfolder0
                 )

if 1:
  if seed==1:
    save_hyperparameters(sim)
  print(f'thermalisation started ; {seed=}')
  start = time()
  sim = thermalisation(sim)
  print(f'thermalisation took {(time()-start)/60:.2f} minutes')

if False:
  start = time()
  sim = annealing(sim)
  print(f'annealing took {(time()-start)/60:.2f} minutes')

sim.model = energy(sim.model)
print(f'final e:{sim.model.E/sim.model.L}')
E_exact = (f_exact_PBC(1/sim.model.T,sim.model.L)+
            sim.model.T * s_exact_PBC(1/sim.model.T,sim.model.L))
print(f'E_exact = {E_exact:.5f}')

export_spins(sim,seed)



