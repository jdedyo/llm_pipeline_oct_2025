#!/bin/bash
# run this file to inference an LLM on OOS chunks
# bash oos.sh LLM_NAME MIN_CHUNK_NUM MAX_CHUNK_NUM
# e.g. bash download_pdfs_range.sh f1 0 10

# Check if at least three arguments are provided
if [ $# -lt 3 ]; then
    echo "Error: Please provide LLM name in the first argument, min oos chunk num in second, and max oos chunk num in third."
    exit 1
fi

llm=$1
minchunk=$2
maxchunk=$3

echo ">> Running $llm jobs for oos chunks between $minchunk and $maxchunk"

# Loop over years
for (( c=$minchunk; c<=$maxchunk; c++ ))
do
    echo "Submitting job for chunk $c"
    sbatch_script="oos.sbatch"  # Change this to the name of your SBATCH script template; will overwrite any job name in the sbatch file
    job_name="oos_${llm}_chunk_${c}"

    sbatch --job-name="$job_name" "$sbatch_script" "$llm" "$c"
done