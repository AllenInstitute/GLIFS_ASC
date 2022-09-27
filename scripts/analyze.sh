#! /bin/bash

# gpu = 1

# echo "Analyzing on $1"

cd ..
cd main/examples

# Create directory for PMNIST
# mkdir -p results
# mkdir -p "results\$1"

# if [ gpu -eq 1 ]
# then
# python3 analyze.py --num_workers "$1" --accelerator gpu --gpus 2 --strategy ddp_find_unused_parameters_false
# else 
python3 analyze.py --num_workers "$1"
# fi
