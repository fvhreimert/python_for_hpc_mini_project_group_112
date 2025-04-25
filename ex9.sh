#!/bin/sh
#BSUB -q c02613
#BSUB -J ex9
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 5
#BSUB -o ex9_%J.out
#BSUB -e ex9_%J.err

# activate env
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# run python script with time command
python ex9.py 20