#!/bin/sh
#BSUB -q c02613
#BSUB -J ex8
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 5
#BSUB -o ex8_%J.out
#BSUB -e ex8_%J.err

# activate env
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# run python script with time command
python ex8.py 20