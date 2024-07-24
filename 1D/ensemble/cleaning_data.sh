#!/bin/bash
L=10000
T=2
formatted_T=$(printf "%.2f" $T)
groupped_spins_filename="T${formatted_T}.txt"

r_idmin=0
r_idmax=50
for (( r_id=r_idmin; r_id<=r_idmax; r_id++ ))
do
  path="/scratch/sacevedo/Ising-Chain/canonical/L$L/r_id$r_id/"
  cd $path
  echo r_id=$r_id
  if [ -e "$groupped_spins_filename" ]; then       # if groupped_spins_filename file is there, I remove the rest of the ungroupped files...
    rm -rf seed*
  else
    echo groupped file missing! 
  fi
done

echo this took: $((SECONDS/60)) m