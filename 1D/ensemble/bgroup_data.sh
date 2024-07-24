#!/bin/bash

mkdir -p log_group

n_seeds=5000
L=10000
r_idmin=56
r_idmax=70

for (( r_id=r_idmin; r_id<=r_idmax; r_id++ ))
do
  sbatch sgroup_data.sh "$n_seeds" "$L" "$r_id"
  sleep 0.01
done
