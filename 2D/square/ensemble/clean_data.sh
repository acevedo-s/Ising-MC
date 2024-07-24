#!/bin/bash
lattice=triangular
L_idmin=3
L_idmax=13
for ((L_id=L_idmin; L_id<=L_idmax; L_id++))
do
  L=$((L_id*10))
  path="/scratch/sacevedo/Ising-$lattice/canonical/L$L/"
  cd $path
  echo L=$L
  find . -name "seed*.npy" -delete
done

echo this took: $((SECONDS/60)) m