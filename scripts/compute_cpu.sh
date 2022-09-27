#! /bin/bash

echo "Training on $1"

cd /gscratch/deepthought/chloewin/GLIFR_review/main/examples

# Create directory for PMNIST
mkdir -p results
mkdir -p "results\$1"

# lhet = "/gscratch/deepthought/chloewin/GLIFR_review/main/examples/results/$1/$x"
# for x in "rnn" "glifr_fheta" "glifr_fhet" "glifr_homa" "glifr_hom" "lstmn" "glifr_lheta" "glifr_lhet" "glifr_rheta" "glifr_rhet"
# do
#    echo "Training $x"
#    mkdir -p "results\$1\$x"
#    python3 train.py --task "$1" --model "$x" --default_root_dir "/gscratch/deepthought/chloewin/GLIFR_review/main/examples/results/$1/$x" --accelerator gpu --devices -1 --strategy ddp_find_unused_parameters_false
# done
# python3 train.py --task "$1" 

python3 train.py --task "$1" --ntrials "$2" --num_workers "$3" 

# python3 train.py --task lmnist --model glifr_lheta --default_root_dir /gscratch/deepthought/chloewin/GLIFR_review/main/results --accelerator gpu --devices -1 --strategy ddp_find_unused_parameters_false