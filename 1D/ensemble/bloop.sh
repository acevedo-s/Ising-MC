#!/bin/bash
#SBATCH --job-name=bloop-CHAIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem=20G
#SBATCH --partition=regular2,regular1
# SBATCH --qos=fastlane # for debugging
# SBATCH --array=1-1
# SBATCH --output=./log_output/%x.o%A-%a   # Standard output
# SBATCH --error=./log_output/%x.o%A-%a    # Standard error
#SBATCH --output=./log_output/%x.o%j
#SBATCH --error=./log_output/%x.o%j 
mkdir -p log_output

seed0=1
n_seeds=$1
L=$2
r_id=$3
echo r_id="$r_id"

for (( seed_id=seed0; seed_id<$((seed0+n_seeds)); seed_id++ ))
do
  seed=$((seed_id + r_id*n_seeds))
  echo seed=$seed
  sbatch smain.sh "$seed" "$L" "$r_id"
  sleep 1
done

