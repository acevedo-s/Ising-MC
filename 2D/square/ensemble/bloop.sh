#!/bin/bash
seed0=3001
n_seeds=2000
L=$1
for (( seed=$seed0; seed<$((seed0+n_seeds)); seed++ ))
do
  echo 'seed='$seed
  sbatch smain.sh $seed $L
  sleep .05
done
