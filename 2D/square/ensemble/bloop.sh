#!/bin/bash
seed0=1
n_seeds=5000
# l_min=8
# l_max=8
l_list=(13)
for (( seed=$seed0; seed<$((seed0+n_seeds)); seed++ ))
do
  # for (( l=$l_min; l<=$l_max; l++ ))
  for l in "${l_list[@]}"
  do
    L=$((10*$l))
    echo seed=$seed,L=$L
    sbatch smain.sh $seed $L
    sleep 6
  done
done
