import os
import sys

# import wandb
from typing import Callable
from jsonargparse import ArgumentParser, namespace_to_dict
from jsonargparse import ActionYesNo
import os
# os.environ["WANDB_SILENT"] = 'true'

from jax.config import config
config.update('jax_platform_name', 'cpu')
# config.update("jax_enable_x64", True)

parser = ArgumentParser(default_config_files=['./sim_config.yaml'])
parser.add_argument('--seed', type=int)
parser.add_argument('--model.J1', type=float)
parser.add_argument('--model.J2', type=float)
parser.add_argument('--model.L', type=int)
parser.add_argument('--model.h', type=float)
parser.add_argument('--mc.T0', type=float)
parser.add_argument('--mc.Tf', type=float)
parser.add_argument('--mc.dT', type=float)
parser.add_argument('--mc.Ntherm0', type=int)
parser.add_argument('--mc.Ntherm', type=int)
parser.add_argument('--mc.Nsamples', type=int)
parser.add_argument('--mc.Nsweeps', type=int)
parser.add_argument('--mc.backend', type=Callable)
cfg = parser.parse_args()