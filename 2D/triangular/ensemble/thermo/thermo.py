import numpy as np
import matplotlib.pyplot as plt
import os

resultsfolder = 'results/'
os.makedirs(resultsfolder,exist_ok=True)
eps = 1E-5
geometry = 'Ising-triang'

def energy(X):
  Ns,L,_ = X.shape
  E = np.zeros(shape=Ns)
  for i in range(L):
    for j in range(L):
      E += X[:,i,j] * ( X[:,(i+1)%L,j]
                       +X[:,i,(j+1)%L]
                       +X[:,(i+1)%L,(j-1)%L]
                       )
  return E

L_list = np.array([32])
T_list = np.arange(0.05,3+eps,.05)
Ns0 = 10000
normalize = 1
# T_list = np.concatenate((np.arange(2.2,2.5+eps,.01),
#                          np.arange(2.6,4+eps,.1)))
for L_id,L in enumerate(L_list):
  N = L**2
  datafolder = f'/scratch/sacevedo/{geometry}/canonical/L{L}/'
  # hyperp = np.loadtxt(datafolder + 'hyperp.txt')
  # Ns = hyperp[6].astype(int)
  E = np.zeros(shape=(Ns0,len(T_list)))
  for T_id,T in enumerate(T_list):
    datafile = datafolder + f'T{T:.2f}.npy'
    X = np.load(f'{datafile}')#[:Ns,:]
    Ns = X.shape[0]
    if T_id==0:print(f'{X.shape=}')
    E[:Ns,T_id] = energy(X)
  E = E[:Ns,:]
  print(f'{E.shape=}')
  E_mean = np.mean(E,axis=0)
  Cv = (np.mean(E**2,axis=0)-E_mean**2) / T_list**2

S = np.array([np.trapz(Cv[:idx]/T_list[:idx],T_list[:idx]) / np.log(2) 
     for idx in range(0,len(T_list))])
if normalize:
  S /= N
  E_mean /= N
  Cv /= N
  E0 = -1
  S0 = .323 / np.log(2)

if True:
  np.savetxt(fname=f'results/thermo.txt',
             X=np.transpose([T_list,S,E_mean,Cv]),fmt='%1.3f')
  fig,ax = plt.subplots(1)
  ax.plot(T_list,E_mean-E0,'o',label=r'$E-E_0/N$')
  ax.plot(T_list,Cv,'o',label='Cv/N')
  ax.plot(T_list,S,'o',label='S/N')
  ax.hlines(S0,T_list[0],T_list[-1],label=r'$S_0(N=\infty)$')
  Tc_id = np.where(np.isclose(Cv,np.max(Cv)))[0][0]
  ax.vlines(T_list[Tc_id],
             plt.ylim()[0],
             plt.ylim()[1],
             color='gray')
  ax.set_xlabel('T')
  ax.legend()
  fig.savefig(fname=f'{resultsfolder}thermo.png')

  


