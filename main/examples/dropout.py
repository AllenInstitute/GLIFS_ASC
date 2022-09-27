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
from models.analyzer import NetworkAnalyzer
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

def train(args, version=0): 
    pct = args.dropout_prob
    print(f"training {args.model} on {args.task} on trial {version} with dropout probability {pct}")
    args.__dict__["default_root_dir"] = os.path.abspath(f"results/{args.task}-dropout{int(pct*100)}/{args.model}/trial_{version}")
    print(f"default root dir used: {args.default_root_dir}")
    # if not os.path.exists(args.default_root_dir):
    os.makedirs(args.default_root_dir, exist_ok=True)
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
    if args.model == "rnn" and args.task in ["lmnist", "sine"]:
        args.__dict__["synaptic_delay"] = 0
    # tb_logger = TensorBoardLogger(str(version))
    # args.__dict__["logger"] = tb_logger
    trainer = pl.Trainer.from_argparse_args(args)
    delattr(args, 'callbacks')
    # delattr(args, 'logger')
    model = model_class(**vars(args))

    # if args.model in ["glifr_rhet", "glifr_rheta", "glifr_fhet", "glifr_fheta"]:
    #     model = model_class.load_from_checkpoint(args.model_ckpt)
    print(f"training {args.model} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    data_module = data_modules[args.task](**vars(args))
    trainer.fit(model, data_module)
    # trainer.save_checkpoint("final.ckpt")
    # trainer.test(datamodule=data_module, ckpt_path=os.path.join(trainer.logger.log_dir, "checkpoints", "last.ckpt"))
    analyzer = NetworkAnalyzer(trainer, model, data_module)
    analyzer.run_ablation_experiment([pct], 10)
    return trainer.logger.log_dir

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

    pcts = [0, 0.2, 0.4,0.6, 0.8, 1]
    for pct in pcts:
        args.__dict__["dropout_prob"] = pct
        model_names = ["rnn", "lstmn", "glifr_lheta", "glifr_fheta", "glifr_rheta", "glifr_homa", "glifr_lhet", "glifr_fhet", "glifr_rhet", "glifr_hom"]
        ckpt_dict = {}
        to_train = model_names#["rnn", "glifr_lheta", "glifr_rheta"]
        for m in ["rnn", "lstmn", "glifr_lheta", "glifr_lhet", "glifr_homa", "glifr_hom"]:
        # for m in ["rnn"]:
            if m not in to_train:
                print(f"{m} not in to_train")
                continue
            args.__dict__["model"] = m
            args.__dict__["dropout_prob"] = pct
            ckpt_dict[m] = train(args)

        for m in ["glifr_fhet", "glifr_rhet"]:
            if m not in to_train:
                print(f"{m} not in to_train")
                continue
            args.__dict__["model"] = m
            args.__dict__["dropout_prob"] = pct
            args.__dict__["ckpt_path"] = os.path.join(ckpt_dict["glifr_lhet"], "checkpoints", "last.ckpt")
            train(args)
            delattr(args, "ckpt_path")

        for m in ["glifr_fheta", "glifr_rheta"]:
            if m not in to_train:
                print(f"{m} not in to_train")
                continue
            args.__dict__["model"] = m
            args.__dict__["dropout_prob"] = pct
            args.__dict__["ckpt_path"] = os.path.join(ckpt_dict["glifr_lheta"], "checkpoints", "last.ckpt")
            train(args)
            delattr(args, "ckpt_path")
