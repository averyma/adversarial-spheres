#!/bin/bash

#SBATCH --mem=4G
#SBATCH -c 2 
#SBATCH --gres=gpu:1 
#SBATCH -p p100 
#SBATCH --output=./slurm_out/slurm_%j.log

python main.py --method $1 --pgd_alpha $2 --pgd_itr $3
