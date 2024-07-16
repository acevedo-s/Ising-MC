import numpy as np 
import os,sys

eps = 1E-6
n_seeds = int(sys.argv[1])
print(f'{n_seeds=}')
L = int(sys.argv[2])
print(f'{L=}')
r_id = int(sys.argv[3])
print(f'{r_id=}')
lattice = 'Chain'
datafolder0 = f'/scratch/sacevedo/Ising-{lattice}/canonical/L{L}/'

T_list = [2]
seed_list = r_id * n_seeds + np.arange(1,n_seeds+1,dtype=int)

datafolder = datafolder0 + f'r_id{r_id}/'  
resultsfolder = datafolder0  + f'/r_id{r_id}/'
for T_id,T in enumerate(T_list):
  print(f'{T=:.2f}')
  X = np.zeros(shape=(n_seeds,L))
  os.makedirs(resultsfolder,exist_ok=True)
  for seed_id,seed in enumerate(seed_list):
    fname = f'{datafolder}seed{seed}/seed{seed}_T{T:.2f}.npy'
    X[seed_id] = np.load(file=fname)
  np.savetxt(resultsfolder+f'T{T:.2f}.txt',
              np.reshape(X,(n_seeds,L))
              )
print(f'finito')
  
