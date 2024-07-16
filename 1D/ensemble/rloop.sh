#!/bin/bash

mkdir -p log_output

n_seeds=$1
L=$2
Nrealizations=75
r_idmin=44

DEPENDENCY_ID=$(sbatch bloop.sh "$n_seeds" "$L" "$r_idmin" | awk '{print $4}')
echo dependency: "$DEPENDENCY_ID"
for (( r_id=r_idmin+1; r_id<Nrealizations; r_id++ ))
do
  DEPENDENCY_ID=$(sbatch --dependency=afterok:"$DEPENDENCY_ID" bloop.sh "$n_seeds" "$L" "$r_id" | awk '{print $4}')
  sleep 0.05
  # sbatch --dependency=afterok:"$DEPENDENCY_ID" sgroup_data.sh "$n_seeds" "$L" "$r_id"
  echo dependency: "$DEPENDENCY_ID"
  sleep 0.05
done
