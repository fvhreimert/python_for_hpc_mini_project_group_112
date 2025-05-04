#!/bin/bash
#BSUB -J parallel_scaling[1-10]       
#BSUB -q hpc
#BSUB -W 45                      
#BSUB -n 18                      
#BSUB -R "rusage[mem=8GB]"       
#BSUB -R "select[model==XeonGold6126]"
#BSUB -o output/static_scale_%J_%I.out 
#BSUB -e output/static_scale_%J_%I.err 
#BSUB -B
#BSUB -N

PYTHON_SCRIPT="simulate_parallel_scaling.py" 
NUM_FLOORPLANS=20

WORKER_ARRAY=(1 2 4 6 8 10 12 14 16 18) 

INDEX=$LSB_JOBINDEX
ARRAY_INDEX=$((INDEX - 1))
NUM_WORKERS=${WORKER_ARRAY[$ARRAY_INDEX]}


echo "Job Array Task Index (LSB_JOBINDEX): $INDEX"
echo "Worker Array Index: $ARRAY_INDEX"
echo "Number of Workers (P) for this task: $NUM_WORKERS"

source /work3/s203520/miniconda/etc/profile.d/conda.sh
conda activate env1

echo "Running ${PYTHON_SCRIPT} for N=${NUM_FLOORPLANS} with P=${NUM_WORKERS} workers..."
time python "${PYTHON_SCRIPT}" "${NUM_FLOORPLANS}" "${NUM_WORKERS}"
echo "Task $INDEX finished."