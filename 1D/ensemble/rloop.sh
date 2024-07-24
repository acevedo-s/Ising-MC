#!/bin/bash

mkdir -p log_output

n_seeds=$1
L=$2
r_idmax=100
r_idmin=74

DEPENDENCY_ID=$(sbatch bloop.sh "$n_seeds" "$L" "$r_idmin" | awk '{print $4}')
echo dependency: "$DEPENDENCY_ID"
for (( r_id=r_idmin+1; r_id<r_idmax; r_id++ ))
do
  DEPENDENCY_ID=$(sbatch --dependency=afterok:"$DEPENDENCY_ID" bloop.sh "$n_seeds" "$L" "$r_id" | awk '{print $4}')
  sleep 0.05
  # sbatch --dependency=afterok:"$DEPENDENCY_ID" sgroup_data.sh "$n_seeds" "$L" "$r_id"
  echo dependency: "$DEPENDENCY_ID"
  sleep 0.05
done
