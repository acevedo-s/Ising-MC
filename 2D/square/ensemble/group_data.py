import numpy as np 
import os,sys

eps = 1E-6
n_seeds = int(sys.argv[1])
print(f'{n_seeds=}')
L = int(sys.argv[2])
print(f'{L=}')
lattice = sys.argv[3]
print(f'{lattice=}')
datafolder0 = f'/scratch/sacevedo/Ising-{lattice}/canonical/'

if lattice ==  'triangular':
  T_list = np.arange(.1,4+eps,0.1)
elif lattice == 'square':
  T_list = np.arange(1,2.1+eps,0.1)
  T_list = np.concatenate((T_list,
                          np.arange(2.2,2.4+eps,.01))
                         )
  T_list = np.concatenate((T_list,
                         np.arange(2.5,4+eps,.1))
                         )

seed_list = range(1,n_seeds+1)
for T_id,T in enumerate(T_list):
  print(f'{T=:.2f}')
  X = np.zeros(shape=(n_seeds,L,L))
  resultsfolder = datafolder0  + f'L{L}/'
  os.makedirs(resultsfolder,exist_ok=True)
  for seed_id,seed in enumerate(seed_list):
    datafolder = datafolder0 + f'L{L}/seed{seed}_'  
    fname = f'{datafolder}T{T:.2f}.npy'
    X[seed_id] = np.load(file=fname)

  np.savetxt(resultsfolder+f'T{T:.2f}.txt',np.reshape(X,(n_seeds,L*L)))
    
