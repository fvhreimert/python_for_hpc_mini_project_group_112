#!/bin/bash
#BSUB -J simulate_original_timing # Job name
#BSUB -q hpc                     
#BSUB -W 30                      
#BSUB -n 1                       
#BSUB -R "rusage[mem=4GB]"       
#BSUB -R "select[model==XeonGold6126]"
#BSUB -o output/simulate_original_%J.out
#BSUB -e output/simulate_original_%J.err
#BSUB -B                         
#BSUB -N                         


PYTHON_SCRIPT_PATH="simulate.py"
NUM_FLOORPLANS=20

source /work3/s203520/miniconda/etc/profile.d/conda.sh
conda activate env1

# --- Run the original Python script and time it ---
echo "Running script: ${PYTHON_SCRIPT_PATH} for ${NUM_FLOORPLANS} floorplans..."
echo "Start time: $(date)"

# Use the 'time' command to measure execution time
time python "${PYTHON_SCRIPT_PATH}" "${NUM_FLOORPLANS}"

echo "End time: $(date)"
echo "Job finished."

# --- Timing Information ---
echo "Timing information will be printed to the .err file by the 'time' command."
echo "Look for 'real', 'user', 'sys' times in output/simulate_original_${LSB_JOBID}.err"