#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH -t 12:00:00
#SBATCH -A berzelius-2024-286
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user shutong@kth.se
#SBATCH -o /home/x_shuji/wasp_rep/logs/slurm-%A.out
#SBATCH -e /home/x_shuji/wasp_rep/logs/slurm-%A.err

module load Anaconda/2023.09-0-hpc1-bdist
conda activate vot

python distill.py -m vae -c ./configs/train_config_vae.yaml