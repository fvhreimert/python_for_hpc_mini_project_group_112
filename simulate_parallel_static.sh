#!/bin/bash
#BSUB -J static_scale[1-7]       # Job name + Array indices (adjust range 1-7 based on WORKER_ARRAY size)
#BSUB -q hpc
#BSUB -W 45                      # Wall time (adjust for N=100, needs to cover the longest run, likely P=1)
#BSUB -n 12                      # <<<<< Max cores needed by ANY task in the array
#BSUB -R "rusage[mem=8GB]"       # Memory per task (adjust if needed)
#BSUB -R "select[model==XeonGold6126]"
#BSUB -o output/static_scale_%J_%I.out # Output file per task (%I is the index)
#BSUB -e output/static_scale_%J_%I.err # Error file per task (contains timing)
#BSUB -B
#BSUB -N

PYTHON_SCRIPT="simulate_parallel_static.py" # Your Python script name
NUM_FLOORPLANS=20

WORKER_ARRAY=(1 2 4 6 8 10 12)

# -- Get worker count based on job array index 
INDEX=$LSB_JOBINDEX
ARRAY_INDEX=$((INDEX - 1))

# Check if the index is valid for the array
if [ $ARRAY_INDEX -lt 0 ] || [ $ARRAY_INDEX -ge ${#WORKER_ARRAY[@]} ]; then
    echo "Error: Job index $INDEX is out of bounds for WORKER_ARRAY." >&2
    exit 1
fi

# Get the number of workers for this specific task
NUM_WORKERS=${WORKER_ARRAY[$ARRAY_INDEX]}
echo "Job Array Task Index (LSB_JOBINDEX): $INDEX"
echo "Worker Array Index: $ARRAY_INDEX"
echo "Number of Workers (P) for this task: $NUM_WORKERS"
# --- --------------------------------------- ---

source /work3/s203520/miniconda/etc/profile.d/conda.sh
conda activate env1


# --- Run the Python script with 'time' ---
echo "Running ${PYTHON_SCRIPT} for N=${NUM_FLOORPLANS} with P=${NUM_WORKERS} workers..."
# The 'time' command will print timing (real, user, sys) to this task's stderr file
# The python script's CSV output goes to this task's stdout file
# The python script's internal timing print goes to this task's stderr file
time python "${PYTHON_SCRIPT}" "${NUM_FLOORPLANS}" "${NUM_WORKERS}"

echo "Task $INDEX finished."