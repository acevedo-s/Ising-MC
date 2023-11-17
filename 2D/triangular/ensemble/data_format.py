import numpy as np 
import os

datafolder0 = f'/scratch/sacevedo/Ising-triang/canonical/'
eps = 1E-6
L = 32
n_seeds = 650
T_list = np.arange(0,3+eps,0.05)

seed_list = range(0,n_seeds)
for T_id,T in enumerate(T_list):
  X = np.zeros(shape=(n_seeds,L,L))
  resultsfolder = datafolder0  + f'L{L}/'
  os.makedirs(resultsfolder,exist_ok=True)
  for seed_id,seed in enumerate(seed_list):
    datafolder = datafolder0 + f'L{L}_seed_{seed}/'  
    fname = f'{datafolder}T{T:.2f}.npy'
    X[seed_id] = np.load(file=fname)

  np.save(resultsfolder+f'T{T:.2f}',X)
    