#!/bin/bash
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:0
#SBATCH --job-name="phi_01"
#SBATCH --output=phi_01.out
#SBATCH --mem=16G 
#SBATCH --nodelist=gpuz01

module load cuda/12.1

module load anaconda
source activate final
export PYTHONPATH=.
model=phi-1_5
python scripts/evaluate_tofu.py --forget_set_name forget01 --model_name ${model} --batch_size 64 --classifier_threshold 0.99 --task_config config/task_config/tofu.yaml --optimal_corrupt_dim 507
