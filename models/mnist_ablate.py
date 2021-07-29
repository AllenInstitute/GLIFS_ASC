import matplotlib
matplotlib.use('Agg')

# author: @chloewin
# 03/07/21
import pickle
import datetime
import utils as ut
from networks import RNNFC, BNNFC
from neurons.glif_new import BNNC, RNNC

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as tud
import torch.nn as nn
import math

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    brnn_name = "final-brnn-initwithbursts-withoutdelay_256units_smnist_linebyline_lateralconns"#"final-brnn-initwithbursts-withdelay_256units_smnist_linebyline_lateralconns"#"brnn-initwithbursts-withdelay_256units_smnist_linebyline_lateralconns"#"brnn-initwithbursts_256units_smnist_linebyline_lateralconns"#"brnn-initwithburst_256units_smnist_linebyline"
    brnn_noasc_name = "final-brnn-initwithbursts-withdelay-noasc_260units_smnist_linebyline_lateralconns"
    brnn_wtonly_name = "final-brnn-initwithbursts-withdelay-notrainparams_264units_smnist_linebyline_lateralconns"
    rnn_name = "final-rnn-initwithbursts-withoutdelay_264units_smnist_linebyline_lateralconns"#"final-rnn-initwithbursts-withoutdelay_264units_smnist_linebyline_lateralconns"#rnn_264units_smnist_linebyline_lateralconns"#"rnn-wodel_103units_smnist_linebyline"

    base_name = "figures_wkof_071821/"
    base_name_save = "traininfo_wkof_071821/"
    base_name_model = "models_wkof_071821/"

    linebyline=True

    dt = 0.05

    pct_remove = 0.2

    hid_size_brnn = 256
    hid_size_brnn_wtonly = 264
    hid_size_brnn_noasc = 260
    hid_size_rnn = 264
    
    input_size = 1
    output_size = 10
    if linebyline:
            input_size = 28

    batch_size = 128
    ntrials = 10

    pcts = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    results_rnn = np.zeros((len(pcts), ntrials))
    results_brnn = np.zeros((len(pcts), ntrials))
    results_brnn_noasc = np.zeros((len(pcts), ntrials))
    results_brnn_wtonly = np.zeros((len(pcts), ntrials))

    model_rnn = RNNFC(in_size = input_size, hid_size = hid_size_rnn, out_size = output_size, dt=dt)
    model_brnn = BNNFC(in_size = input_size, hid_size = hid_size_brnn, out_size = output_size, ascs=True, learnparams=True)
    model_brnn_wtonly = BNNFC(in_size = input_size, hid_size = hid_size_brnn_wtonly, out_size = output_size, ascs=True, learnparams=False)
    model_brnn_noasc = BNNFC(in_size = input_size, hid_size = hid_size_brnn_noasc, out_size = output_size, ascs=False, learnparams=True)

    print(f"rnn: {count_parameters(model_rnn)}")
    print(f"brnn: {count_parameters(model_brnn)}")
    print(f"brnn_wtonly: {count_parameters(model_brnn_wtonly)}")
    print(f"brnn_noasc: {count_parameters(model_brnn_noasc)}")

    for i in range(len(pcts)):
        pct_remove = pcts[i]
        for j in range(ntrials):
            model_rnn = RNNFC(in_size = input_size, hid_size = hid_size_rnn, out_size = output_size, dt=dt)
            model_brnn = BNNFC(in_size = input_size, hid_size = hid_size_brnn, out_size = output_size, ascs=True, learnparams=True)
            model_brnn_wtonly = BNNFC(in_size = input_size, hid_size = hid_size_brnn_wtonly, out_size = output_size, ascs=True, learnparams=False)
            model_brnn_noasc = BNNFC(in_size = input_size, hid_size = hid_size_brnn_noasc, out_size = output_size, ascs=False, learnparams=True)
            model_rnn.load_state_dict(torch.load("saved_models/" + base_name_model + rnn_name + ".pt"))
            model_brnn.load_state_dict(torch.load("saved_models/" + base_name_model + brnn_name + ".pt"))
            model_brnn_wtonly.load_state_dict(torch.load("saved_models/" + base_name_model + brnn_wtonly_name + ".pt"))
            model_brnn_noasc.load_state_dict(torch.load("saved_models/" + base_name_model + brnn_noasc_name + ".pt"))

            idx_brnn = np.random.choice(hid_size_brnn, int(pct_remove * hid_size_brnn), replace=False)
            idx_brnn_wtonly = np.random.choice(hid_size_brnn_wtonly, int(pct_remove * hid_size_brnn_wtonly), replace=False)
            idx_brnn_noasc = np.random.choice(hid_size_brnn_noasc, int(pct_remove * hid_size_brnn_noasc), replace=False)
            idx_rnn = np.random.choice(hid_size_rnn, int(pct_remove * hid_size_rnn), replace=False)

            model_rnn.silence(idx_rnn)
            model_brnn.silence(idx_brnn)
            model_brnn_wtonly.silence(idx_brnn_wtonly)
            model_brnn_noasc.silence(idx_brnn_noasc)

            # Train model
            num_epochs = 0
            lr = 0.001#1e-8#0.001#0.0025#0.0025#25#1#25
            reg_lambda = 1500

            training_info_brnn = ut.train_rbnn_mnist(model_brnn, batch_size, num_epochs, lr, not False, verbose = False,linebyline=linebyline)
            #training_info_brnn_noasc = ut.train_rbnn_mnist(model_brnn_noasc, batch_size, num_epochs, lr, not False, verbose = False,linebyline=linebyline, ascs=False)
            #training_info_brnn_wtonly = ut.train_rbnn_mnist(model_brnn_wtonly, batch_size, num_epochs, lr, not False, verbose = False,linebyline=linebyline, trainparams=False)
            #training_info_rnn = ut.train_rbnn_mnist(model_rnn, batch_size, num_epochs, lr, not True, verbose = False,linebyline=linebyline)#, output_text_filename="results_ablate_" + str(pct_remove) + "_" + rnn_name + "_repeat.txt")
            
            #results_rnn[i,j] = training_info_rnn["test_accuracy"]
            results_brnn[i,j] = training_info_brnn["test_accuracy"]
            #results_brnn_wtonly[i,j] = training_info_brnn_wtonly["test_accuracy"]
            #results_brnn_noasc[i,j] = training_info_brnn_noasc["test_accuracy"]

    #np.savetxt("mnist_results/results_" + rnn_name + ".csv", results_rnn, delimiter=",")
    np.savetxt("mnist_results/results_" + brnn_name + "_nodelay.csv", results_brnn, delimiter=",")
    #np.savetxt("mnist_results/results_" + brnn_wtonly_name + ".csv", results_brnn_wtonly, delimiter=",")
    #np.savetxt("mnist_results/results_" + brnn_noasc_name + ".csv", results_brnn_noasc, delimiter=",")

if __name__ == '__main__':
        main()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
