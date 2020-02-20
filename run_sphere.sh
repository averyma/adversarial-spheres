touch /checkpoint/ama/${SLURM_JOB_ID}/DELAYPURGE
CHECKPOINTS_DIR=/checkpoint/ama/${SLURM_JOB_ID}

#python -u main_sphere.py --method reg_1st --seed 0 --pgd_itr 10 --lambbda 1 \
#--checkpoint_dir $CHECKPOINTS_DIR \
#--checkpoint_freq 10000 \

#python -u main_sphere.py --method reg_1st --seed 1 --pgd_itr 10 --lambbda 1 \
#--checkpoint_dir $CHECKPOINTS_DIR \
#--checkpoint_freq 10000 \

#python -u main_sphere.py --method reg_1st --seed 2 --pgd_itr 10 --lambbda 1 \
#--checkpoint_dir $CHECKPOINTS_DIR \
#--checkpoint_freq 10000 \

#python -u main_sphere.py --method reg_2nd --seed 0 --pgd_itr 2 --lambbda 1 \
#--checkpoint_dir $CHECKPOINTS_DIR \
#--checkpoint_freq 10000 \

#python -u main_sphere.py --method reg_2nd --seed 1 --pgd_itr 2 --lambbda 1 \
#--checkpoint_dir $CHECKPOINTS_DIR \
#--checkpoint_freq 10000 \

#python -u main_sphere.py --method reg_2nd --seed 2 --pgd_itr 2 --lambbda 1 \
#--checkpoint_dir $CHECKPOINTS_DIR \
#--checkpoint_freq 10000 \


