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
                 i=0,
                 L=None,
                 ):
        self.key = key
        self.spins = spins
        self.E = E
        self.dE = dE
        self.T = T
        self.h = h
        self.i = i
        self.L = L
    def _tree_flatten(self):
        children = (self.key,
                    self.spins,
                    self.E,self.dE,
                    self.T,
                    self.h,
                    self.i,
                    )  # arrays / dynamic values
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
    model.spins = 2*jax.random.randint(subkey, (model.L,model.L), minval=0, maxval=2)-1
    ### GS energy check:
    # model.spins = jnp.ones(shape=(model.L,model.L)).astype(jnp.int32)
    # for i in range(0,model.L,1):
    #    for j in range(0,model.L,2):
    #      model.spins = model.spins.at[i,j].set(-1)
    # print(model.spins)
    model = energy(model)
    return model

@jax.jit
def _energy_fix_i(j,model):
  model.E += model.spins[model.i,j]*(
             model.spins[(model.i+1)%model.L,j]
            +model.spins[model.i,(j+1)%model.L]
            +model.spins[(model.i+1)%model.L,(j-1)%model.L]
          )
  return model

@jax.jit
def _energy(i,model):
  model.i = i
  model = jax.lax.fori_loop(lower=0,
                upper=model.L,
                body_fun=_energy_fix_i,
                init_val=model)
  return model

def energy(model):
  model.E = 0
  model = jax.lax.fori_loop(lower=0,
                upper=model.L,
                body_fun=_energy,
                init_val=model)
  return model
