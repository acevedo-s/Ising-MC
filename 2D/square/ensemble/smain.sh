#!/bin/bash
#SBATCH --job-name=MC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=12G
#SBATCH --partition=regular2,regular1
# SBATCH --qos=fastlane # for debugging
# SBATCH --array=1-1
# SBATCH --output=./log_output/%x.o%A-%a   # Standard output
# SBATCH --error=./log_output/%x.o%A-%a    # Standard error
#SBATCH --output=./log_output/%x.o%j
#SBATCH --error=./log_output/%x.o%j 
python3 -u main.py $1 $2 # seed, L