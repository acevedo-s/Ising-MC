import sys,os
seed = int(sys.argv[1])
print(f'{seed=}')
L = int(sys.argv[2])
print(f'{L=}')
lattice = 'triangular'
print(f'{lattice=}')

T0 = 4
Tf = .1
dT = .1
Ntherm0 = 5000
Ntherm = 2000
print(f'{T0=:.3f}')

if lattice == 'square':
  square = 1
  triangular = 0
elif lattice == 'triangular':
  square = 0
  triangular = 1