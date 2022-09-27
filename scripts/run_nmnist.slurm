#!/bin/bash

#SBATCH --job-name=run-pmnist
#SBATCH --partition=gpu-2080ti
#SBATCH --account=deepthought
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
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

chmod +x compute.sh
./compute.sh nmnist 10 8