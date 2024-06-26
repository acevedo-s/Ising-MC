import sys,os
import numpy as np
import sys,os
from time import time

seed = int(sys.argv[1])
print(f'{seed=}')
L = int(sys.argv[2])
print(f'{L=}')
r_id = int(sys.argv[3])
print(f'{r_id=}')

lattice = 'Ising-Chain'
T0 = 2
Tf = 2
dT = 9999
Ntherm0 = 5000
Ntherm = 0

print(f'{T0=:.3f}')