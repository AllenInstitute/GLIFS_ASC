import matplotlib
matplotlib.use('Agg')
"""
This file trains a network of rate-based GLIF neurons with after-spike currents on a sequential MNIST task.
It tests the procedure on multiple random initializations.

For each iteration i (0 indexed), membrane parameters are saved to
f"results/{base_name_results}-{hid_size}units-{i}itr-init-membraneparams.csv" before training
f"results/{base_name_results}-{hid_size}units-{i}itr-membraneparams.csv" after training
where the first column lists thresh and the second column lists k_m

For each iteration i (0 indexed), after-spike current parameters are saved to
f"results/{base_name_results}-{hid_size}units-{i}itr-init-ascparams.csv" before training
f"results/{base_name_results}-{hid_size}units-{i}itr-ascparams.csv" after training
where the first column lists k_j, the second column lists r_j, and the third column lists a_j.

For each iteration i (0-indexed), the CrossEntropy loss on each epoch is saved to
f"results/{base_name_results}-{hid_size}units-{i}itr-losses.csv"

For each iteration i (0-indexed), the accuracies on the testing set when different percentages of
neurons are ablated are saved to
f"results/{base_name_results}-{hid_size}units-{i}itr-ablation.csv"
where each row corresponds to a number of trials using a single percentage of neurons silenced.

For each iteration i (0-indexed), the accuracies on the testing set for the number of trials are saved to
f"results/{base_name_results}-{hid_size}units-accs.csv"

For each iteration i (0-indexed), the PyTorch model dictionary is saved to
f"saved_models/{base_name_model}-{hid_size}units-{i}itr-init.pt" before training
f"saved_models/{base_name_model}-{hid_size}units-{i}itr.pt" after training

For each iteration i (0-indexed), the dictionary of parameters, gradients, etc. collected
over training is stored to
f"traininfo/{base_name_traininfo}-itr.pickle"

NOTE: Requires a data folder populated with MNIST data.
"""

import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

import utils_train as utt
import utils_misc as utm
from models.networks import LSTMFC, RNNFC, BNNFC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Base Filename")
    parser.add_argument("condition", help="One of ['rnn', 'lstm', 'glifr-hominit', 'glifr-hetinit']")
    parser.add_argument("learnparams", type=int, help="0 or 1 whether to learn parameters")
    parser.add_argument("numascs", type=int, help="Number of ASCs")
    parser.add_argument("anneal", type=int, help="whether to anneal sigma_v", default=0)
 
    args = parser.parse_args()
    learnparams = (args.learnparams == 1)
    anneal = (args.anneal == 1)
 
    main_name = args.name
    base_name_traininfo = "smnist_extra/" + main_name
    base_name_model = "smnist_extra/" + main_name
    base_name_results = "smnist_extra/" + main_name

    in_size = 28
    out_size = 10
    num_params = utm.count_params_glif(in_size=28, hid_size=256,out_size=10, num_asc=2, learnparams=True)
    if args.condition == "lstm":
        hid_size = utm.hid_size_lstm(num_params=num_params, in_size=in_size, out_size=out_size) 
        print(utm.count_params_lstm(in_size=in_size, hid_size=hid_size, out_size=out_size))
    elif args.condition == "rnn":
        hid_size = utm.hid_size_rnn(num_params=num_params, in_size=in_size, out_size=out_size)
        print(utm.count_params_rnn(in_size=in_size, hid_size=hid_size, out_size=out_size))
    elif args.condition == "glifr-hetinit":
        hid_size = utm.hid_size_glif(num_params=num_params, in_size=in_size, out_size=out_size, learnparams=learnparams, num_asc = args.numascs)
        print(utm.count_params_glif(in_size=in_size, hid_size=hid_size, out_size=out_size, num_asc = args.numascs, learnparams=learnparams))
    elif args.condition == "glifr-hominit":
        hid_size = utm.hid_size_glif(num_params=num_params, in_size=in_size, out_size=out_size, learnparams=learnparams, num_asc = args.numascs)
        print(utm.count_params_glif(in_size=in_size, hid_size=hid_size, out_size=out_size, num_asc = args.numascs, learnparams=learnparams))

    # Model parameters
    ascs = (args.numascs > 0)
    dt = 0.05
    num_ascs = args.numascs
    rnn_delay = 1

    # Training parameters
    batch_size = 128
    num_epochs = 50
    lr = 0.001
    itrs = 30
    sgd = False

    # Experiment parameters
    pcts = [0,0.2,0.4,0.6,0.8,1.0]
    ntrials = 30

    # Gather results
    accs = []

    for i in range(itrs):
        if args.condition == "rnn":
            print("using rnn")
            model = RNNFC(in_size = in_size, hid_size = hid_size, out_size = out_size, dt=dt, delay=rnn_delay)
        elif args.condition == "lstm":
            print("using lstm")
            model = LSTMFC(in_size = in_size, hid_size = hid_size, out_size = out_size, dt=dt)
        else:
            print("using glifr")
            hetinit = (args.condition == "glifr-hetinit")
            print(f"hetinit: {hetinit}; learnparams: {learnparams}")
            model = BNNFC(in_size = in_size, hid_size = hid_size, out_size = out_size, dt=dt, hetinit=hetinit, ascs=ascs, learnparams=learnparams)

        print(f"using {utm.count_parameters(model)} parameters and {hid_size} neurons")
        torch.save(model.state_dict(), "saved_models/" + base_name_model + "-" + str(hid_size) + "units-" + str(i) + "itr-init.pt")

        # Record parameters of initialized network
        if args.condition[0:5] == "rglif":
            membrane_parameters = np.zeros((hid_size, 2))
            membrane_parameters[:, 0] = model.neuron_layer.thresh.detach().numpy().reshape(-1)
            membrane_parameters[:, 1] = model.neuron_layer.transform_to_k(model.neuron_layer.trans_k_m).detach().numpy().reshape(-1)
            np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(i) + "itr-init-membraneparams.csv", membrane_parameters, delimiter=',')
            
            if ascs:
                asc_parameters = np.zeros((hid_size * num_ascs, 3))
                asc_parameters[:, 0] = model.neuron_layer.transform_to_k(model.neuron_layer.trans_asc_k)[:,0,:].detach().numpy().reshape(-1)
                asc_parameters[:, 1] = model.neuron_layer.transform_to_asc_r(model.neuron_layer.trans_asc_r)[:,0,:].detach().numpy().reshape(-1)
                asc_parameters[:, 2] = model.neuron_layer.asc_amp[:,0,:].detach().numpy().reshape(-1)
                np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(i) + "itr-init-ascparams.csv", asc_parameters, delimiter=',')
        print(f"Training on iteration {i}")
        
        # Train network
        training_info = utt.train_rbnn_mnist(model, batch_size, num_epochs, lr, args.condition[0:5] == "glifr", verbose = True, trainparams=learnparams, linebyline=True, ascs=ascs, sgd=sgd, anneal=anneal)

        torch.save(model.state_dict(), "saved_models/" + base_name_model + "-" + str(hid_size) + "units-" + str(i) + "itr.pt")
        np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(i) + "itr-losses.csv", np.array(training_info["losses"]), delimiter=',')
        
        # Record parameters of trained network
        if args.condition[0:5] == "rglif":
            membrane_parameters = np.zeros((hid_size, 2))
            membrane_parameters[:, 0] = model.neuron_layer.thresh.detach().numpy().reshape(-1)
            membrane_parameters[:, 1] = model.neuron_layer.transform_to_k(model.neuron_layer.trans_k_m).detach().numpy().reshape(-1)
            np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(i) + "itr-membraneparams.csv", membrane_parameters, delimiter=',')

            if ascs:
                asc_parameters = np.zeros((hid_size * num_ascs, 3))
                asc_parameters[:, 0] = model.neuron_layer.transform_to_k(model.neuron_layer.trans_asc_k)[:,0,:].detach().numpy().reshape(-1)
                asc_parameters[:, 1] = model.neuron_layer.transform_to_asc_r(model.neuron_layer.trans_asc_r)[:,0,:].detach().numpy().reshape(-1)
                asc_parameters[:, 2] = model.neuron_layer.asc_amp[:,0,:].detach().numpy().reshape(-1)
                np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(i) + "itr-ascparams.csv", asc_parameters, delimiter=',')

        # Record performance after random silencing
        model.eval()
        ablation_results = np.zeros((len(pcts), ntrials))
        for pct_idx in range(len(pcts)):
            pct_remove = pcts[pct_idx]
            for trial_idx in range(ntrials):
                print(f"Training on iteration {i}......Trial {trial_idx}...Ablating {pct_remove}")
                idx = np.random.choice(hid_size, int(pct_remove * hid_size), replace=False)
                model.silence(idx)
                training_info_silence = utt.train_rbnn_mnist(model, batch_size, 0, lr, args.condition[0:5] == "rglif", verbose = False, trainparams=learnparams,linebyline=True, ascs=ascs, sgd=sgd, anneal=anneal)
                ablation_results[pct_idx, trial_idx] = training_info_silence["test_accuracy"]
        np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(i) + "itr-ablation.csv", ablation_results, delimiter=',')

        accs.append(training_info["test_accuracy"])

        if i % 2 == 0:
            with open("traininfo/" + base_name_traininfo + "-" + str(i) + "itr.pickle", 'wb') as handle:
                pickle.dump(training_info, handle)
    
    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "accs.csv", np.array(accs), delimiter=",")

if __name__ == '__main__':
        main()
