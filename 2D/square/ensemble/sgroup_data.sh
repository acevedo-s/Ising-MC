#!/bin/bash
#SBATCH --job-name=Grpdata
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=10G
#SBATCH --partition=regular2,regular1
# SBATCH --qos=fastlane # for debugging
# SBATCH --array=1-1
# SBATCH --output=./log_output/%x.o%A-%a   # Standard output
# SBATCH --error=./log_output/%x.o%A-%a   # Standard error
#SBATCH --output=./log_output/%x.o%j
#SBATCH --error=./log_output/%x.o%j 
n_seeds=$1
L=$2
lattice=$3
python3 -u group_data.py $n_seeds $L $lattice