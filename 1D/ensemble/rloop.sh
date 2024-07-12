#!/bin/bash

mkdir -p log_output

n_seeds=$1
L=$2
Nrealizations=22
r_idmin=18

FIRST_JOB_ID=$(sbatch bloop.sh "$n_seeds" "$L" "$r_idmin" | awk '{print $4}')
echo dependency: "$FIRST_JOB_ID"
for (( r_id=r_idmin+1; r_id<Nrealizations; r_id++ ))
do
  FIRST_JOB_ID=$(sbatch --dependency=afterok:"$FIRST_JOB_ID" bloop.sh "$n_seeds" "$L" "$r_id" | awk '{print $4}')
  # sbatch --dependency=afterok:"$FIRST_JOB_ID" sgroup_data.sh "$n_seeds" "$L" "$r_id"
  echo dependency: "$FIRST_JOB_ID"
  sleep 0.05
done
