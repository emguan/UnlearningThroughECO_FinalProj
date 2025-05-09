#!/bin/bash
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:0
#SBATCH --job-name="demo_eco"
#SBATCH --output=demo.out
#SBATCH --mem=16G


# Load CUDA module (modify version if needed)
module load cuda/12.1

# Activate your conda environment
conda activate final  # or conda activate final

# Run your Python script
python /home/cs601-eguan3/601.471_FinalProj_Sp25/llm-unlearn-eco/demo_copy.py

