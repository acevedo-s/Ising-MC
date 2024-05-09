#!/bin/bash
seed0=4001
n_seeds=1000
l_min=3
l_max=10
for (( seed=$seed0; seed<$((seed0+n_seeds)); seed++ ))
do
  for (( l=$l_min; l<=$l_max; l++ ))
  do
    L=$((10*$l))
    echo seed=$seed,L=$L
    sbatch smain.sh $seed $L
    sleep 7
  done
done
