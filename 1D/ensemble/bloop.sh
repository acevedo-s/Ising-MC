#!/bin/bash

mkdir -p log_output

seed0=1
n_seeds=5000
L=10000
Nrealizations=1

for (( seed=seed0; seed<$((seed0+n_seeds)); seed++ ))
do
  for (( r_id=0; r_id<Nrealizations; r_id++ ))
  do
    echo seed=$seed,L=$L,r_id=$r_id
    sbatch smain.sh $seed $L "$r_id"
    # job_id=$(sbatch your_script.sh | awk '{print $4}')
    sleep 0.01
  done
done
