import matplotlib
matplotlib.use('Agg')

# author: @chloewin
# 03/07/21
import argparse
import pickle
import datetime
import utils as ut
import utils_train as utt
import utils_misc as utm
from networks import LSTMFC, RNNFC, BNNFC
from neurons.glif_new import BNNC, RNNC

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.optim as optim
import torch.utils.data as tud
import torch.nn as nn
import math
# import torch.utils.data.DataLoader

#torch.autograd.set_detect_anomaly(True)
"""
This file trains a network of rate-based GLIF neurons with after-spike currents on a pattern generation task.
1. Single pattern generation: generate a sinusoid of a given frequency when provided with constant input
2. Sussillo pattern generation: generate a sinusoid of a freuqency that is proportional to the amplitude of the constant input
3. Bellec pattern generation: generation a sinusoid of a frequency that corresponds to the subset of input neurons receiving input

Trained model is saved to the folder specified by model_name + date.
Figures on learned outputs, parameters, weights, gradients, and losses over training are saved to the folder specified by fig_name + date

Loss is printed on every epoch

To alter model architecture, change sizes, layers, and conns dictionaries. 
There are other specifications including amount of time, number of epochs, learning rate, etc.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Base Filename")
    parser.add_argument("condition", help="One of ['rnn', 'lstm', 'rglif-hominit', 'rglif-hetinit']")
    parser.add_argument("learnparams", type=int, help="0 or 1 whether to learn parameters")
    parser.add_argument("numascs", type=int, help="Number of ASCs")
 
    args = parser.parse_args()
    learnparams = (args.learnparams == 1)
 
    main_name = args.name
    # base_name = "figures_wkof_072521/" + main_name
    base_name_traininfo = "traininfo_wkof_080821/" + main_name
    base_name_model = "models_wkof_080821/" + main_name
    base_name_results = "results_wkof_080821/" + main_name

    in_size = 28
    out_size = 10
    num_params = utm.count_params_glif(in_size=28, hid_size=256,out_size=10, num_asc=2, learnparams=True)
    if args.condition == "lstm":
        hid_size = utm.hid_size_lstm(num_params=num_params, in_size=in_size, out_size=out_size) 
        print(utm.count_params_lstm(in_size=in_size, hid_size=hid_size, out_size=out_size))
    elif args.condition == "rnn":
        hid_size = utm.hid_size_rnn(num_params=num_params, in_size=in_size, out_size=out_size)
        print(utm.count_params_rnn(in_size=in_size, hid_size=hid_size, out_size=out_size))
    elif args.condition == "rglif-hetinit":
        hid_size = utm.hid_size_glif(num_params=num_params, in_size=in_size, out_size=out_size, learnparams=learnparams, num_asc = args.numascs)
        print(utm.count_params_glif(in_size=in_size, hid_size=hid_size, out_size=out_size, num_asc = args.numascs, learnparams=learnparams))
    elif args.condition == "rglif-hominit":
        hid_size = utm.hid_size_glif(num_params=num_params, in_size=in_size, out_size=out_size, learnparams=learnparams, num_asc = args.numascs)
        print(utm.count_params_glif(in_size=in_size, hid_size=hid_size, out_size=out_size, num_asc = args.numascs, learnparams=learnparams))

    #learnparams = (args.condition == "rglif")
    ascs = (args.numascs > 0)
    #initburst = False
    dt = 0.05
    sparseness = 0
    num_ascs = args.numascs

    k_folds = 5
    num_epochs = 40#0
    # doppio/americano [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
    # others [1e-2, 1e0, 1e2, 1e4]
    regs = [1e-50, 1e-30, 1e-10, 1e0]#[1e-12, 1e-9, 1e-6, 1e-3, 1e0, 1e3]#[1e-1, 1e0, 1e1, 1e2, 1e3]#[1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    if args.condition == "rnn":
        regs = [0, 1e-30, 1e-27, 1e-24]
    results = np.zeros((len(regs), k_folds))

    torch.manual_seed(42)
    kfold = KFold(n_splits = k_folds, shuffle = True)
    batch_size = 128
    lr = 0.001
    sgd = False#True

    pcts = [0,0.2,0.4,0.6,0.8,1.0]
    ntrials = 30

    accs = []

    root = './data/mnist'
    trainloader, testloader = utt.mnist_generator(root, batch_size)
    dataset = trainloader.dataset

    for r in range(len(regs)):
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            print(f"FOLD {fold} -- on {r}th reg of {regs[r]}")
            print("----------")
            
            if args.condition == "rnn":
                print("using rnn")
                model = RNNFC(in_size = in_size, hid_size = hid_size, out_size = out_size, dt=dt, sparseness=sparseness)
            elif args.condition == "lstm":
                print("using lstm")
                model = LSTMFC(in_size = in_size, hid_size = hid_size, out_size = out_size, dt=dt)
            else:
                print("using glifr")
                hetinit = (args.condition == "rglif-hetinit")
                print(f"hetinit: {hetinit}; learnparams: {learnparams}")
                model = BNNFC(in_size = in_size, hid_size = hid_size, out_size = out_size, dt=dt, hetinit=hetinit, ascs=ascs, learnparams=learnparams, sparseness=sparseness)

            print(f"using {utm.count_parameters(model)} parameters and {hid_size} neurons")

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size, sampler=test_subsampler)

            training_info = utt.train_rbnn_mnist(model, batch_size, num_epochs, lr, args.condition[0:5] == "rglif", verbose = True, trainparams=learnparams,linebyline=True, ascs=ascs, sgd=sgd, trainloader=trainloader, testloader=testloader, reg_lambda=regs[r])

            print(f"accuracy for fold {fold}: {training_info['test_accuracy']}")
            print("-----------------------------------------------")
            results[r, fold] = training_info["test_accuracy"]

    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "kfold.csv", results, delimiter=",")

if __name__ == '__main__':
        main()
