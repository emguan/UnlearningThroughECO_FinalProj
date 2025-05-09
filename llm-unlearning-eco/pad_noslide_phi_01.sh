#!/bin/bash
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:0
#SBATCH --job-name="pad_no_01"
#SBATCH --output=pad_noslide_phi_01.out
#SBATCH --mem=16G 
#SBATCH --nodelist=gpuz01

source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh

module load cuda/12.1

module load anaconda
conda activate final
export PYTHONPATH=.
model=phi-1_5
CUDA_LAUNCH_BLOCKING=1 python scripts/evaluate_tofu_padded.py --forget_set_name forget01 --model_name ${model} --batch_size 64 --classifier_threshold 0.99 --task_config config/task_config/tofu.yaml --optimal_corrupt_dim 507
