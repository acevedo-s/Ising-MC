#!/bin/bash
L=10000
seed_id0=2501
seed_id1=5000
n_seeds=5000

r_idmin=47
r_idmax=75
for (( r_id=r_idmin+1; r_id<r_idmax; r_id++ ))
do
  for (( seed_id=seed_id0; seed_id<=seed_id1; seed_id++ ))
  do
    seed=$((seed_id + r_id*n_seeds))
    filename="/scratch/sacevedo/Ising-Chain/canonical/L$L/r_id$r_id/seed$seed/"
    if [ -z "$(ls -A $filename)" ]
    then
      sbatch smain.sh "$seed" "$L" "$r_id"
      # echo $seed
    fi
  done
done