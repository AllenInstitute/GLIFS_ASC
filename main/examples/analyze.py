from argparse import ArgumentParser
from copy import copy
from collections import defaultdict
import sys
import yaml

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch.utils.data as tud
import torch

from datasets.sine import SineDataModule
from datasets.mnist import MNISTDataModule
from datasets.nmnist import NMNISTDataModule
from datasets.utils import add_data_args
from models.pl_modules import add_structure_args, add_general_model_args, add_training_args, GLIFRN, RNN, LSTMN
from models.analyzer import NetworkAnalyzer
from training.callbacks import *

# Must specify default_root_dir
data_modules = {
    "sine": SineDataModule,
    "pmnist": MNISTDataModule,
    "lmnist": MNISTDataModule,
    "nmnist": NMNISTDataModule
}

past_ckpts = {
    "sine": {
        "rnn": "./results/sine/rnn/trial_0/lightning_logs/version_0",
        "lstmn": "./results/sine/lstmn/trial_0/lightning_logs/version_0",
        "glifr_lheta": "./results/sine/glifr_lheta/trial_0/lightning_logs/version_0",
        "glifr_lhet": "./results/sine/glifr_lhet/trial_0/lightning_logs/version_0",
        "glifr_rheta": "./results/sine/glifr_rheta/trial_0/lightning_logs/version_0",
        "glifr_rhet": "./results/sine/glifr_rhet/trial_0/lightning_logs/version_0",
        "glifr_fheta": "./results/sine/glifr_fheta/trial_0/lightning_logs/version_0",
        "glifr_fhet": "./results/sine/glifr_fhet/trial_0/lightning_logs/version_0",
        "glifr_homa": "./results/sine/glifr_homa/trial_0/lightning_logs/version_0",
        "glifr_hom": "./results/sine/glifr_hom/trial_0/lightning_logs/version_0"
    },
    "pmnist": {
        "rnn": "",
        "lstmn": "",
        "glifr_lheta": "",
        "glifr_lhet": "",
        "glifr_rheta": "",
        "glifr_rhet": "",
        "glifr_fheta": "",
        "glifr_fhet": "",
        "glifr_homa": "",
        "glifr_hom": ""
    },
    "lmnist": {
        "rnn": "./results/lmnist/rnn/trial_0/lightning_logs/version_4749741",
        "lstmn": "./results/lmnist/lstmn/trial_0/lightning_logs/version_4728746",
        "glifr_lheta": "./results/lmnist/glifr_lheta/trial_0/lightning_logs/version_4728746",
        "glifr_lhet": "./results/lmnist/glifr_lhet/trial_0/lightning_logs/version_4728746",
        "glifr_rheta": "./results/lmnist/glifr_rheta/trial_0/lightning_logs/version_4728746",
        "glifr_rhet": "./results/lmnist/glifr_rhet/trial_0/lightning_logs/version_4728746",
        "glifr_fheta": "./results/lmnist/glifr_fheta/trial_0/lightning_logs/version_4728746",
        "glifr_fhet": "./results/lmnist/glifr_fhet/trial_0/lightning_logs/version_4728746",
        "glifr_homa": "./results/lmnist/glifr_homa/trial_0/lightning_logs/version_4728746",
        "glifr_hom": "./results/lmnist/glifr_hom/trial_0/lightning_logs/version_4728746"
    },
    "nmnist": {
        "rnn": "./results/nmnist/rnn/trial_0/lightning_logs/version_0",
        "lstmn": "./results/nmnist/lstmn/trial_0/lightning_logs/version_0",
        "glifr_lheta": "./results/nmnist/glifr_lheta/trial_0/lightning_logs/version_0",
        "glifr_lhet": "./results/nmnist/glifr_lhet/trial_0/lightning_logs/version_0",
        "glifr_rheta": "./results/nmnist/glifr_rheta/trial_0/lightning_logs/version_0",
        "glifr_rhet": "./results/nmnist/glifr_rhet/trial_0/lightning_logs/version_0",
        "glifr_fheta": "./results/nmnist/glifr_fheta/trial_0/lightning_logs/version_0",
        "glifr_fhet": "./results/nmnist/glifr_fhet/trial_0/lightning_logs/version_0",
        "glifr_homa": "./results/nmnist/glifr_homa/trial_0/lightning_logs/version_0",
        "glifr_hom": "./results/nmnist/glifr_hom/trial_0/lightning_logs/version_0"
    },
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

def analyze(args, version): 
    print(f"training {args.model} on {args.task} on trial {version}")

    args.__dict__["default_root_dir"] = os.path.abspath(f"results/{args.task}-post/{args.model}/trial_{version}")
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
    if args.model == "rnn" and args.task in ["sine", "lmnist"]:
        args.__dict__["synaptic_delay"] = 0
    # tb_logger = TensorBoardLogger(str(version))
    # args.__dict__["logger"] = tb_logger
    trainer = pl.Trainer.from_argparse_args(args)
    if hasattr(args,'callbacks'):
        delattr(args, 'callbacks')
    if hasattr(args,'ckpt_path'):
        delattr(args, 'ckpt_path')
    if hasattr(args,'default_root_dir'):
        delattr(args, 'default_root_dir')
    model = model_class.load_from_checkpoint(os.path.join(past_ckpts[args.task][args.model], "checkpoints/last.ckpt"))#(**vars(args))
    args.__dict__["num_workers"] = model.hparams.num_workers
    args.__dict__["duration_timestep"] = model.hparams.duration_timestep
    args.__dict__["initialization"] = model.hparams.initialization
    args.__dict__["nmnist_path"] = model.hparams.nmnist_path
    args.__dict__["params_learned"] = model.hparams.params_learned
    args.__dict__["num_ascs"] = model.hparams.num_ascs
    # model.hparams.default_root_dir = args.__dict__["default_root_dir"]
    model.hparams.num_workers = args.__dict__["num_workers"]
    data_module = data_modules[args.task](**vars(args))
    data_module.setup()
    # print(f"{data_module.hparams.num_workers} vs {args.__dict__['num_workers']} vs {model.hparams.num_workers}")
    # trainer.fit(model, data_module)
    # trainer.test(datamodule=data_module, ckpt_path=os.path.join(trainer.logger.log_dir, "checkpoints", "last.ckpt"))

    return NetworkAnalyzer(trainer, model, data_module)

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
    # parser.add_argument("--ntrials", type=int, default=1)

    # add model specific args
    args = parser.parse_args()

    # TODO: ablation study for all networks and all tasks
    # TODO: get parameters and f-I curves for LHetA for all tasks
    model_names = ["rnn", "lstmn", "glifr_lheta", "glifr_fheta", "glifr_rheta", "glifr_homa", "glifr_lhet", "glifr_fhet", "glifr_rhet", "glifr_hom"]

    pcts = [0, 0.2, 0.4, 0.6, 0.8, 1]
    
    for t in model_names:
        args.__dict__["task"] = t
        for m in model_names:
            args.__dict__["model"] = m
            analyzer = analyze(args, 0)
            if t == "sine":
                print(f"Running test responses save on {m} for {t}")
                analyzer.test_responses()
            print(f"Running ablation experiment on {m} for {t}")
            # if t not in ["sine"]:
            analyzer.run_ablation_experiment(pcts, ntrials=10)
            if m in ["glifr_lheta", "glifr_fheta"]:
                print(f"Running parameter save on {m} for {t}")
                analyzer.save_parameters()
                analyzer.plot_ficurve()
            