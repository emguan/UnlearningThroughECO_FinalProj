#!/bin/bash
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:0
#SBATCH --job-name="phi_05"
#SBATCH --output=phi_05.out
#SBATCH --mem=16G

module load cuda/12.1

module load anaconda
source activate final

export PYTHONPATH=.
model=phi-1_5
python -m scripts.evaluate_tofu --forget_set_name forget05 --model_name ${model} --batch_size 64 --classifier_threshold 0.99 --task_config config/task_config/tofu.yaml --optimal_corrupt_dim 766
