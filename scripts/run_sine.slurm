#!/bin/bash

#SBATCH --job-name=run-sine
#SBATCH --partition=gpu-2080ti
#SBATCH --account=deepthought
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#SBATCH --gres=gpu:0
#SBATCH --gpus-per-node=0
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chloewin@uw.edu

# Modules to use (optional).

# Your programs to run.
source ~/.bashrc
conda activate /gscratch/deepthought/chloewin/glifr_env

cat $0
echo "--------------------"

chmod +x compute_cpu.sh
./compute_cpu.sh sine 10 1