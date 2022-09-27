from argparse import ArgumentParser
from copy import copy
from collections import defaultdict
import sys
import yaml

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch.utils.data as tud
import torch
# torch.autograd.set_detect_anomaly(True)

from datasets.sine import SineDataModule
from datasets.mnist import MNISTDataModule
from datasets.nmnist import NMNISTDataModule
from datasets.utils import add_data_args
from models.pl_modules import add_structure_args, add_general_model_args, add_training_args, GLIFRN, RNN, LSTMN
from training.callbacks import *

# Must specify default_root_dir
data_modules = {
    "sine": SineDataModule,
    "pmnist": MNISTDataModule,
    "lmnist": MNISTDataModule,
    "nmnist": NMNISTDataModule
}

def read_yaml_into_args(args, filename, modeltype):
    with open(filename) as f:
        data = yaml.load(f, Loader=yaml.loader.SafeLoader)
    # data = yaml.load(filename)
    arg_dict = args.__dict__
    for key, value in data.items():
        if isinstance(value, dict):
            arg_dict[key] = value[modeltype]
            # print(value[modeltype])
        # if key in arg_dict:
        #     continue
        elif isinstance(value, list):
            arg_dict[key] = []
            for v in value:
                arg_dict[key].append(v)
        else:
            arg_dict[key] = value
    return args

def print_params(args): 
    print(f"training {args.model} on {args.task}")

    if args.model == "rnn":
        args = read_yaml_into_args(args, "./../config/rnn.yaml", args.model)
        model_class = RNN
    elif args.model == "lstmn":
        args = read_yaml_into_args(args, "./../config/lstmn.yaml", args.model)
        model_class = LSTMN
    elif args.model.startswith("glifr"):
        args = read_yaml_into_args(args, "./../config/glifr.yaml", args.model)
        model_class = GLIFRN

        # Learning params
        if "lhet" in args.model or "rhet" in args.model:
            args.__dict__["params_learned"] = True
        else:
            args.__dict__["params_learned"] = False
        # Number of ascs
        if not args.model.endswith("a"):
            args.__dict__["num_ascs"] = 0
        # Initialization
        if "fhet" in args.model or "rhet" in args.model:
            args.__dict__["initialization"] = "het"
        else:
            args.__dict__["initialization"] = "hom"
    
    args = read_yaml_into_args(args, f"./../config/{args.task}_task.yaml", args.model)

    # add all the available trainer options to argparse
    # ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
    args.__dict__["callbacks"] = [eval(c)() for c in args.__dict__["callbacks"]]

    model = model_class(**vars(args))
    
    print(f"training {args.model} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

if __name__ == '__main__':
    print(f"found {torch.cuda.device_count()} devices")
    parser = ArgumentParser()
    pl.Trainer.add_argparse_args(parser)

    # Model-specific
    add_structure_args(parser)
    add_general_model_args(parser)
    add_training_args(parser)
    add_data_args(parser)
    RNN.add_model_specific_args(parser)
    LSTMN.add_model_specific_args(parser)
    GLIFRN.add_model_specific_args(parser)

    # Dataset-specific
    SineDataModule.add_sine_args(parser)
    NMNISTDataModule.add_nmnist_args(parser)
    MNISTDataModule.add_mnist_args(parser)

    # add program level args
    parser.add_argument("--task", type=str, choices=["sine", "nmnist", "pmnist", "lmnist"])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ntrials", type=int, default=1)
    # parser.add_argument("--model", type=str, choices=["rnn", "lstmn", "glifr_lheta", "glifr_fheta", "glifr_rheta", "glifr_homa", "glifr_lhet", "glifr_fhet", "glifr_rhet", "glifr_hom"])

    # add model specific args
    args = parser.parse_args()

    model_names = ["rnn", "lstmn", "glifr_lheta", "glifr_lhet", "glifr_homa", "glifr_hom"]
    for m in model_names:
        args.__dict__["model"] = m
        print_params(args)
