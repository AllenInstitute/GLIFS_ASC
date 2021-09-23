import matplotlib
matplotlib.use('Agg')

# author: @chloewin
# 03/07/21
# reviewed @chloewin 09/12/21
import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

import utils_train as utt
import utils_task as utta
import utils_misc as utm
from networks import LSTMFC, RNNFC, BNNFC


"""
This file trains a network of rate-based GLIF neurons with after-spike currents on a pattern generation task.
Sussillo pattern generation: generate a sinusoid of a freuqency that is proportional to the amplitude of the constant input

Trained models are saved to the folder specified by base_name_model.
Accuracies and parameters are saved to the folder specified by base_name_results.
Torch dictionaries for networks along with losses over epochs
are saved to the folder specified by base_name_traininfo.
Loss is printed on every epoch
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Base Filename")
    parser.add_argument("condition", help="One of ['rnn', 'lstm', 'rglif-hetinit', 'rglif-hominit']")
    parser.add_argument("learnparams", type=int, help="Learn parameters?")
    parser.add_argument("numascs", type=int, help="Number of ASCs")
 
    args = parser.parse_args()
 
    main_name = args.name
    base_name = "figures_wkof_080821/" + main_name
    base_name_traininfo = "traininfo_wkof_080821/" + main_name
    base_name_model = "models_wkof_080821/" + main_name
    base_name_results = "results_wkof_080821/" + main_name

    learnparams = (args.learnparams == 1)
    print(learnparams)

    in_size = 1
    out_size = 1
    num_params = utm.count_params_glif(in_size=1, hid_size=128,out_size=1, num_asc=2, learnparams=True)
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

    ascs = (args.numascs > 0)
    dt = 0.05
    sparseness = 0
    num_ascs = args.numascs

    batch_size = 2
    num_epochs = 5000
    lr = 0.0001
    itrs = 10
    sgd = False

    sim_time = 5
    num_freqs = 6
    freq_min = 0.08#0.001
    freq_max = 0.6
    amp = 1
    noise_mean = 0
    noise_std = 0

    freqs = 10 ** np.linspace(np.log10(freq_min), np.log10(freq_max), num=num_freqs)

    pcts = [0,0.2,0.4,0.6,0.8,1.0]
    ntrials = 30

    accs = []

    for i in range(itrs):
        inputs, targets = utta.create_sines_amp(sim_time, dt, amp, noise_mean, noise_std, freqs)
        traindataset = utta.create_dataset(inputs, targets)

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
        torch.save(model.state_dict(), "saved_models/" + base_name_model + "-" + str(hid_size) + "units-" + str(i) + "itr-init.pt")
        
        training_info = utt.train_rbnn(model, traindataset, batch_size, num_epochs, lr, glifr = args.condition[0:5] == "rglif", task = "pattern", decay=False, sgd=sgd, trainparams=learnparams, ascs=ascs)

        torch.save(model.state_dict(), "saved_models/" + base_name_model + "-" + str(hid_size) + "units-" + str(i) + "itr.pt")
        np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(i) + "itr-losses.csv", np.array(training_info["losses"]), delimiter=',')
        
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

        # ablation studies
        with torch.no_grad():
            ablation_results = np.zeros((len(pcts), ntrials))
            for pct_idx in range(len(pcts)):
                pct_remove = pcts[pct_idx]
                for trial_idx in range(ntrials):
                    idx = np.random.choice(hid_size, int(pct_remove * hid_size), replace=False)
                    model.silence(idx)
                    training_info_silence = utt.train_rbnn(model, traindataset, batch_size, 0, lr, glifr = args.condition[0:5] == "rglif", task = "pattern", decay=False, sgd=sgd, trainparams=learnparams, ascs=ascs)
                    ablation_results[pct_idx, trial_idx] = training_info_silence["test_loss"]
            np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(i) + "itr-ablation.csv", ablation_results, delimiter=',')

        colors = ["sienna", "gold", "chartreuse", "darkgreen", "lightseagreen", "deepskyblue", "blue", "darkorchid", "plum", "darkorange", "fuchsia", "tomato", "cyan", "greenyellow", "cornflowerblue", "limegreen", "springgreen", "green", "lightgreen", "aquamarine", "springgreen", "green", "lightgreen", "aquamarine", "springgreen", "green", "lightgreen", "aquamarine", "springgreen", "green", "lightgreen", "aquamarine", "springgreen", "green", "lightgreen"]

        final_outputs_driven = training_info["final_outputs_driven"]
        for i in range(num_freqs):
            plt.plot(np.arange(len(final_outputs_driven[i][0])) * dt, final_outputs_driven[i][0,:,0], c = colors[i % len(colors)], label=f"freq {freqs[i]}")
            plt.plot(np.arange(len(final_outputs_driven[i][0])) * dt, targets[:, i], '--', c = colors[i % len(colors)])
        # plt.legend()
        plt.xlabel("time (ms)")
        plt.ylabel("firing rate (1/ms)")
        plt.savefig("figures/" + base_name + "_final_outputs")
        plt.close()

        np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(i) + "itr-finaloutputs.csv", np.stack(final_outputs_driven).reshape((-1, num_freqs)), delimiter=',')
        np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(i) + "itr-targets.csv", targets.reshape((-1, num_freqs)), delimiter=',')

        accs.append(training_info["test_loss"])

        if i % 2 == 0:
            with open("traininfo/" + base_name_traininfo + "-" + str(i) + "itr.pickle", 'wb') as handle:
                pickle.dump(training_info, handle)
    
    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "accs.csv", np.array(accs), delimiter=",")

if __name__ == '__main__':
        main()
