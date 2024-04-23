import jax
from mc import *
from params import *

eps = 1E-6

resultsfolder0 = f'/scratch/sacevedo/Ising-{lattice}/canonical/L{L}/'  

key0 = jax.random.PRNGKey(seed)
spins0 = jax.numpy.zeros(shape=(L,L),dtype=int)

model = Model(seed=seed,
              key=key0,
              spins=spins0,
              T=T0,
              L=L,
              )
print(f'{T0=:.3f}')
model = init_state(model)
print(f'e0={model.E/L/L}')
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
  start = time.time()
  sim = thermalisation(sim)
  print(f'thermalisation took {(time.time()-start)/60:.2f} minutes')

if 1:
  start = time.time()
  sim = annealing(sim)
  print(f'annealing took {(time.time()-start)/60:.2f} minutes')

sim.model = energy(sim.model)
print('E_final=',sim.model.E/L/L)
print(f'm={sim.model.spins.sum()/L**2}')
# if b == 0 and h==0:
#   E = (L* f_exact_PBC(1/T,L)+
#        T*L*s_exact_PBC(1/T,L))
#   print(f'E_exact = {E:.5f}')
#   Es = energy_chain_PBC(states,normalized=0)
#   meanE = np.mean(Es)
#   stdE = np.std(Es)
#   print(f'<E>={meanE:.3f}+-{stdE:.3f}')




