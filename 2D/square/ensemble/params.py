import sys,os
seed = int(sys.argv[1])
# seed = int(os.environ['SLURM_ARRAY_TASK_ID'])
L = int(sys.argv[2])
lattice = 'square'

T0 = 4
Tf = .1
dT = .1
Ntherm0 = 5000
Ntherm = 2000


# class Model():
#   def __init__(self,
#                L=None,
#                lattice=None):
#     self.L = L
#     self.lattice = lattice

# class Mc():
#   def __init__(self,
#     T0=None,
#     Tf=None,# don't put to 0
#     dT=None,
#     Ntherm0=None,
#     Ntherm=None,
#     Nsamples=None,
#     Nsweeps=None,
#     ):
#     self.T0 = T0
#     self.Tf = Tf
#     self.dT = dT
#     self.Ntherm0 = Ntherm0
#     self.Ntherm = Ntherm
#     self.Nsamples = Nsamples
#     self.Nsweeps = Nsweeps
