import os

def read_last_seed(path):
  with open(path) as f:
    return int(f.readline().split()[-1])

# min_seed = 151
# n_seeds = 2
# max_seed = min_seed + n_seeds
# seed_list = list(range(min_seed,max_seed+1,2))

seed_list = range(0,10000,1)

for seed in seed_list:
  old_seed = read_last_seed('sim_config.yaml')
  os.system(f'sed -i -e \'s/seed: {old_seed}/seed: {seed}/g\' sim_config.yaml')
  os.system(f' python main.py')
