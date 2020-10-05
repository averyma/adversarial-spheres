#!/bin/bash

#SBATCH --mem=4G
#SBATCH -c 2 
#SBATCH --gres=gpu:1 
#SBATCH -p p100 
#SBATCH --output=./slurm_out/slurm_%j.log

JOB_ID=${SLURM_JOB_ID}
echo $JOB_ID

python main.py --method $1 --job_id $JOB_ID --pgd_eps $2 --pgd_itr $3 --lambbda $4
