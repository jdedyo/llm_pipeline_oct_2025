#!/bin/bash
# run this file to train an LLM
# bash train.sh LLM
# e.g. bash process_pdfs_one_year.sh g1


# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    echo "Error: Please provide the LLM to train in the first argument."
    exit 1
fi

if [ $# -gt 0 ]; then
    llm=$1
fi

echo ">> Training $llm."

sbatch_script="train.sbatch"  # Change this to the name of your SBATCH script template; will overwrite any job name in the sbatch file
job_name="train_$llm"

sbatch --job-name="$job_name" "$sbatch_script" $llm

echo "Train job for $llm submitted successfully."