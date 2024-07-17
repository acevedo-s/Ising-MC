#!/bin/bash

mkdir -p log_group

n_seeds=$1
L=$2
r_idmax=22
r_idmin=44

for (( r_id=r_idmin; r_id<=r_idmax; r_id++ ))
do
  sbatch sgroup_data.sh "$n_seeds" "$L" "$r_id"
  sleep 0.01
done
