#!/usr/bin/bash
#BSUB -J ex7
#BSUB -q hpc
#BSUB -W 30
#BSUB -R "rusage[mem=4GB]"
#BSUB -o ex7_%J.out
#BSUB -e ex7_%J.err
#BSUB -R "select[model==XeonGold6126]"
#BSUB -R "span[hosts=1]" 
#BSUB -n 2

# activate env
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# run python script with time command
python ex7.py 20
