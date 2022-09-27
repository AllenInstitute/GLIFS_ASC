#! /bin/bash

# declare -i gpu
gpu=1

echo "Training on $1"

cd ..
cd main/examples

# Create directory for PMNIST
mkdir -p results
# mkdir -p "results\$1"

# if [ gpu -eq 1 ]
# then
python3 dropout.py --task "$1" --ntrials "$2" --num_workers "$3" --accelerator gpu --gpus 2 --strategy ddp_find_unused_parameters_false
# else 
    # python3 train.py --task "$1" --ntrials "$2" --num_workers "$3"
# fi
