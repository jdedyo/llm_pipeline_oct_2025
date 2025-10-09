#!/bin/bash
#SBATCH --job-name=train_llm
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_llm/%x-%j.log
#SBATCH --open-mode=append
#SBATCH --partition=gpu_h200
#SBATCH --gpus=2

module purge
module load miniconda
conda activate pensions401k_llm_sep_2025

export PYTHONUNBUFFERED=1
python train.py --model="$1"