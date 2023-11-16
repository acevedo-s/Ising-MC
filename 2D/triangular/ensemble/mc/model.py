import jax 
import jax.numpy as jnp
from jax.experimental.host_callback import call


class Model:
    def __init__(self, 
                 key=None, 
                 spins=None,
                 E=None,         # energy of spins
                 dE=0,
                 T=None,
                 h=None,
                 L=None,):
        self.key = key
        self.spins = spins
        self.E = E
        self.dE = dE
        self.T = T
        self.h = h
        self.L = L
    def _tree_flatten(self):
        children = (self.key,
                    self.spins,
                    self.E,self.dE,
                    self.T,
                    self.h,)  # arrays / dynamic values
        aux_data = {'L': self.L}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
jax.tree_util.register_pytree_node(Model,
                               Model._tree_flatten,
                               Model._tree_unflatten)

@jax.jit
def init_state(model):
    model.key, subkey = jax.random.split(model.key, num=2)
    model.spins = 2*jax.random.randint(subkey, (model.L,), minval=0, maxval=2)-1
    model = energy(model)
    call(lambda x: print(f''),1)
    call(lambda x: print(f'L={x}'),model.L)
    return model

@jax.jit
def _energy(i,model):
  model.E -= model.spins[i]*model.spins[(i+1)%model.L]
  return model

@jax.jit
def energy(model):
  model.E = 0
  model = jax.lax.fori_loop(lower=0,
                upper=model.L,
                body_fun=_energy,
                init_val=model)
  return model