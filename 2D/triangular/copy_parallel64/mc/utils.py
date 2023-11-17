import numpy as np

def s_exact_PBC(beta, L):
  """
  Exact entropy per site of the 1D ferromagnetic Ising chain with
  periodic boundary conditions and all couplings equal to 1
  """
  partial_beta_log_Z = np.tanh(beta) + 1/(np.sinh(beta) * np.cosh(beta)) * \
                    (1/(1+np.cosh(beta)/np.sinh(beta)))**L 
  log_Z = np.log(2) + np.log(np.cosh(beta)) + (1/L) * np.log(1 + (np.tanh(beta))**L)
  return  - beta * partial_beta_log_Z + log_Z

def f_exact_PBC(beta, L):
    """
    Exact free energy per site of the 1D ferromagnetic Ising chain with
    periodic boundary conditions and all couplings equal to 1
    """
    return (-1/beta) * (np.log(2) + np.log(np.cosh(beta)) + (1/L) * np.log(1 + np.tanh(beta)**L))

def energy_chain_PBC(X,normalized):
  """shape of X is (Nsamples,Nspins)"""
  L = np.shape(X)[1]
  if normalized:
    X = 2 * X - 1
  return - np.sum([ X[:,i] * X[:,(i+1)%L] for i in range(L)],axis=0)