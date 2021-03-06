import matplotlib
matplotlib.use('Agg')

"""
This file trains a network of rate-based GLIF neurons with after-spike currents on a sequential MNIST task.
It tests the procedure on multiple random initializations where the network is subject
to dropout during training (one random initialization per dropout percentage). 
Random subsets of neurons are then silenced during inference.

For each iteration with percentage pct, membrane parameters are saved to
f"results/{base_name_results}-{hid_size}units-{pct}ablated-init-membraneparams.csv" before training
f"results/{base_name_results}-{hid_size}units-{pct}ablated-membraneparams.csv" after training
where the first column lists thresh and the second column lists k_m

For each iteration with percentage pct, after-spike current parameters are saved to
f"results/{base_name_results}-{hid_size}units-{pct}ablated-init-ascparams.csv" before training
f"results/{base_name_results}-{hid_size}units-{pct}ablated-ascparams.csv" after training
where the first column lists k_j, the second column lists r_j, and the third column lists a_j.

For each iteration with percentage pct, the CrossEntropy loss on each epoch is saved to
f"results/{base_name_results}-{hid_size}units-{pct}ablated-losses.csv"

The accuracies on the testing set for the number of trials are saved to
f"results/{base_name_results}-{hid_size}units-ablated-accs.csv"
where each row corresponds to a different dropout probability and the columns represent the multiple
random subsets chosen to be ablated during inference for that given trained network.

For each iteration with percentage pct, the PyTorch model dictionary is saved to
f"saved_models/{base_name_model}-{hid_size}units-{pct}ablated-init.pt" before training
f"saved_models/{base_name_model}-{hid_size}units-{pct}ablated.pt" after training

NOTE: Requires a data folder populated with MNIST data.
"""

import argparse

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
 
    args = parser.parse_args()
    learnparams = (args.learnparams == 1)
 
    main_name = args.name
    base_name_model = "test/" + main_name
    base_name_results = "test/" + main_name

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

    # Training parameters
    batch_size = 128
    num_epochs = 50
    lr = 0.001
    sgd = False

    # Experiment parameters
    pcts = [0,0.2,0.4,0.6,0.8,1.0]
    nablation = 30

    results = np.zeros((len(pcts), nablation))

    # Gather results
    for j in range(len(pcts)):
        pct = pcts[j]
        if args.condition == "rnn":
            print("using rnn")
            model = RNNFC(in_size = in_size, hid_size = hid_size, out_size = out_size, dt=dt, dropout_prob=pct)
        elif args.condition == "lstm":
            print("using lstm")
            model = LSTMFC(in_size = in_size, hid_size = hid_size, out_size = out_size, dt=dt, dropout_prob=pct)
        else:
            print("using glifr")
            hetinit = (args.condition == "glifr-hetinit")
            print(f"hetinit: {hetinit}; learnparams: {learnparams}")
            model = BNNFC(in_size = in_size, hid_size = hid_size, out_size = out_size, dt=dt, hetinit=hetinit, ascs=ascs, learnparams=learnparams, dropout_prob=pct)

        print(f"using {utm.count_parameters(model)} parameters and {hid_size} neurons")
        torch.save(model.state_dict(), "saved_models/" + base_name_model + "-" + str(hid_size) + "units-" + str(pct) + "ablated-" + "init.pt")

        if args.condition[0:5] == "glifr":
            membrane_parameters = np.zeros((hid_size, 2))
            membrane_parameters[:, 0] = model.neuron_layer.thresh.detach().numpy().reshape(-1)
            membrane_parameters[:, 1] = model.neuron_layer.transform_to_k(model.neuron_layer.trans_k_m).detach().numpy().reshape(-1)
            np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(pct) + "ablated-" + "init-membraneparams.csv", membrane_parameters, delimiter=',')
            
            if ascs:
                asc_parameters = np.zeros((hid_size * num_ascs, 3))
                asc_parameters[:, 0] = model.neuron_layer.transform_to_k(model.neuron_layer.trans_asc_k)[:,0,:].detach().numpy().reshape(-1)
                asc_parameters[:, 1] = model.neuron_layer.transform_to_asc_r(model.neuron_layer.trans_asc_r)[:,0,:].detach().numpy().reshape(-1)
                asc_parameters[:, 2] = model.neuron_layer.asc_amp[:,0,:].detach().numpy().reshape(-1)
                np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(pct) + "ablated-" + "init-ascparams.csv", asc_parameters, delimiter=',')
        print(f"Training on pct {j}")
        training_info = utt.train_rbnn_mnist(model, batch_size, num_epochs, lr, args.condition[0:5] == "glifr", verbose = True, trainparams=learnparams, linebyline=True, ascs=ascs, sgd=sgd)#, output_text_filename = "results/" + base_name_results + "_" + str(i) + "itr_performance.txt")

        torch.save(model.state_dict(), "saved_models/" + base_name_model + "-" + str(hid_size) + "units-" + str(pct) + "ablated" + ".pt")
        np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(pct) + "ablated-losses.csv", np.array(training_info["losses"]), delimiter=',')
        
        if args.condition[0:5] == "glifr":
            membrane_parameters = np.zeros((hid_size, 2))
            membrane_parameters[:, 0] = model.neuron_layer.thresh.detach().numpy().reshape(-1)
            membrane_parameters[:, 1] = model.neuron_layer.transform_to_k(model.neuron_layer.trans_k_m).detach().numpy().reshape(-1)
            np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(pct) + "ablated-" + "membraneparams.csv", membrane_parameters, delimiter=',')

            if ascs:
                asc_parameters = np.zeros((hid_size * num_ascs, 3))
                asc_parameters[:, 0] = model.neuron_layer.transform_to_k(model.neuron_layer.trans_asc_k)[:,0,:].detach().numpy().reshape(-1)
                asc_parameters[:, 1] = model.neuron_layer.transform_to_asc_r(model.neuron_layer.trans_asc_r)[:,0,:].detach().numpy().reshape(-1)
                asc_parameters[:, 2] = model.neuron_layer.asc_amp[:,0,:].detach().numpy().reshape(-1)
                np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(pct) + "ablated-" + "ascparams.csv", asc_parameters, delimiter=',')

        for i in range(nablation):
            idx = np.random.choice(hid_size, int(pct * hid_size), replace=False)
            model.silence(idx)
            training_info_silence = utt.train_rbnn_mnist(model, batch_size, 0, lr, args.condition[0:5] == "glifr", verbose = False, trainparams=learnparams,linebyline=True, ascs=ascs, sgd=sgd)
        
            results[j, i] = training_info_silence["test_accuracy"]
    
        np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-ablated-" + "accs.csv", results, delimiter=",")

if __name__ == '__main__':
        main()
