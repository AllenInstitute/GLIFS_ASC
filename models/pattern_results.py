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
    parser.add_argument("condition", help="One of ['rnn', 'lstm', 'rglif', 'rglif_wtonly']")
    parser.add_argument("numascs", type=int, help="Number of ASCs")
 
    args = parser.parse_args()
 
    main_name = args.name
    # base_name = "figures_wkof_072521/" + main_name
    base_name_traininfo = "traininfo_wkof_080121/" + main_name
    base_name_model = "models_wkof_080121/" + main_name
    base_name_results = "results_wkof_080121/" + main_name

    in_size = 1
    out_size = 1
    num_params = utm.count_params_glif(in_size=1, hid_size=128,out_size=1, num_asc=2, learnparams=True)
    if args.condition == "lstm":
        hid_size = utm.hid_size_lstm(num_params=num_params, in_size=in_size, out_size=out_size) 
        print(utm.count_params_lstm(in_size=in_size, hid_size=hid_size, out_size=out_size))
    elif args.condition == "rnn":
        hid_size = utm.hid_size_rnn(num_params=num_params, in_size=in_size, out_size=out_size)
        print(utm.count_params_rnn(in_size=in_size, hid_size=hid_size, out_size=out_size))
    elif args.condition == "rglif_wtonly":
        hid_size = utm.hid_size_glif(num_params=num_params, in_size=in_size, out_size=out_size, learnparams=False, num_asc = args.numascs)
        print(utm.count_params_glif(in_size=in_size, hid_size=hid_size, out_size=out_size, num_asc = args.numascs, learnparams=False))
    elif args.condition == "rglif":
        hid_size = utm.hid_size_glif(num_params=num_params, in_size=in_size, out_size=out_size, learnparams=True, num_asc = args.numascs)
        print(utm.count_params_glif(in_size=in_size, hid_size=hid_size, out_size=out_size, num_asc = args.numascs, learnparams=True))

    learnparams = (args.condition == "rglif")
    ascs = (args.numascs > 0)
    initburst = False
    dt = 0.05
    sparseness = 0
    num_ascs = args.numascs

    batch_size = 2
    num_epochs = 5000
    lr = 0.0001
    itrs = 10
    sgd = True

    sim_time = 5
    num_freqs = 6
    freq_min = 0.08#0.001
    freq_max = 0.6
    amp = 1
    noise_mean = 0
    noise_std = 0

    freqs = 10 ** np.linspace(np.log10(freq_min), np.log10(freq_max), num=num_freqs)

    pcts = [0,0.2,0.4,0.6,0.8,1.0]
    ntrials = 10

    accs = []

    for i in range(itrs):
        inputs, targets = utt.create_sines_amp(sim_time, dt, amp, noise_mean, noise_std, freqs)
        traindataset = utt.create_dataset(inputs, targets)

        if args.condition == "rnn":
            print("using rnn")
            model = RNNFC(in_size = in_size, hid_size = hid_size, out_size = out_size, dt=dt, sparseness=sparseness)
        elif args.condition == "lstm":
            print("using lstm")
            model = LSTMFC(in_size = in_size, hid_size = hid_size, out_size = out_size, dt=dt)
        else:
            print("using glifr")
            model = BNNFC(in_size = in_size, hid_size = hid_size, out_size = out_size, dt=dt, initburst=initburst, ascs=ascs, learnparams=learnparams, sparseness=sparseness)

        print(f"using {utm.count_parameters(model)} parameters and {hid_size} neurons")

        training_info = ut.train_rbnn(model, traindataset, batch_size, num_epochs, lr, glifr = args.condition[0:5] == "rglif", task = "pattern", decay=False, sgd=sgd, trainparams=learnparams, ascs=ascs)

        torch.save(model.state_dict(), "saved_models/" + base_name_model + "-" + str(hid_size) + "units-" + str(i) + "itr.pt")
        np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(i) + "itr-losses.csv", np.array(training_info["losses"]), delimiter=',')
        
        if args.condition[0:5] == "rglif":
            membrane_parameters = np.zeros((hid_size, 2))
            membrane_parameters[:, 0] = model.neuron_layer.thresh.detach().numpy().reshape(-1)
            membrane_parameters[:, 1] = model.neuron_layer.transform_to_k(model.neuron_layer.trans_k_m).detach().numpy().reshape(-1)
            np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(i) + "itr-membraneparams.csv", membrane_parameters, delimiter=',')

            asc_parameters = np.zeros((hid_size * num_ascs, 3))
            asc_parameters[:, 0] = model.neuron_layer.transform_to_k(model.neuron_layer.trans_asc_k)[:,0,:].detach().numpy().reshape(-1)
            asc_parameters[:, 1] = model.neuron_layer.transform_to_asc_r(model.neuron_layer.trans_asc_r)[:,0,:].detach().numpy().reshape(-1)
            asc_parameters[:, 2] = model.neuron_layer.asc_amp[:,0,:].detach().numpy().reshape(-1)
            np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(i) + "itr-ascparams.csv", asc_parameters, delimiter=',')

        # ablation studies
        ablation_results = np.zeros((len(pcts), ntrials))
        for pct_idx in range(len(pcts)):
            pct_remove = pcts[pct_idx]
            for trial_idx in range(ntrials):
                idx = np.random.choice(hid_size, int(pct_remove * hid_size), replace=False)
                model.silence(idx)
                training_info_silence = ut.train_rbnn(model, traindataset, batch_size, 0, lr, glifr = args.condition[0:5] == "rglif", task = "pattern", decay=False, sgd=sgd, trainparams=learnparams, ascs=ascs)
                ablation_results[pct_idx, trial_idx] = training_info_silence["test_loss"]
        np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(i) + "itr-ablation.csv", ablation_results, delimiter=',')

        accs.append(training_info["test_accuracy"])

        if i % 2 == 0:
            with open("traininfo/" + base_name_traininfo + "-" + str(i) + "itr.pickle", 'wb') as handle:
                pickle.dump(training_info, handle)
    
    print(accs)
    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "accs.csv", np.array(accs), delimiter=",")

if __name__ == '__main__':
        main()
