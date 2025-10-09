#!/bin/bash
#SBATCH --job-name=test_llm
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --output=logs/test_llm/%x-%j.log
#SBATCH --open-mode=append
#SBATCH --partition=gpu_h200
#SBATCH --gpus=2

module purge
module load miniconda
conda activate pensions401k_llm_sep_2025

export PYTHONUNBUFFERED=1
python test.py --model="$1"