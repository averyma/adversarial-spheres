#!/bin/bash

#SBATCH --mem=4G
#SBATCH -c 2 
#SBATCH --gres=gpu:1 
#SBATCH -p p100 
#SBATCH --output=./slurm_out/slurm_%j.log

JOB_ID=${SLURM_JOB_ID}
echo $JOB_ID

python main.py --method $1 --pgd_alpha $2 --pgd_itr $3 --optim $4 --job_id $JOB_ID --total_itrs 100000
