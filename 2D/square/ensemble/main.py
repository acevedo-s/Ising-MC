import jax
from mc import *
from params import *

eps = 1E-6

resultsfolder0 = f'/scratch/sacevedo/Ising-{lattice}/canonical/L{L}/'  

key0 = jax.random.PRNGKey(seed)
spins0 = jax.numpy.zeros(shape=(L,L),dtype=int)
samples0 = jax.numpy.zeros(shape=(Nsamples,L,L),dtype=int)

key0, subkey = jax.random.split(key0, num=2)

model = Model(key=key0,
              spins=spins0,
              T=T0,
              L=L,
              # h=h,
              )
print(f'{T0=:.3f}')
model = init_state(model)
print(model.E/L/L)
sim = Simulation(samples=samples0,
                 model=model,
                 Tf=Tf,
                 dT=dT,
                 Ntherm=Ntherm0,
                 Nsamples=Nsamples,
                 Nsweeps=Nsweeps,
                 resultsfolder=resultsfolder0
                 )
sys.exit()

if 1:
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
print('E_final=',sim.E/L/L)
print(sim.spins)
# if b == 0 and h==0:
#   E = (L* f_exact_PBC(1/T,L)+
#        T*L*s_exact_PBC(1/T,L))
#   print(f'E_exact = {E:.5f}')
#   states = sim.samples
#   Es = energy_chain_PBC(states,normalized=0)
#   meanE = np.mean(Es)
#   stdE = np.std(Es)
#   print(f'<E>={meanE:.3f}+-{stdE:.3f}')




