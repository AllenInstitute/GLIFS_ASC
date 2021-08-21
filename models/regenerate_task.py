import matplotlib as mpl
mpl.use('Agg')

# author: @chloewin
# 03/07/21
import pickle
import datetime
import utils as ut
import utils_train as utt
from networks import RNNFC, BNNFC
from neurons.glif_new import BNNC, RNNC
from pylab import cm


import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as tud
import torch.nn as nn
import math
import random
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
    font = {'family' : 'arial',
        # 'weight' : 'bold',
        'size'   : 22}

    mpl.rc('font', **font)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # ax.set_xlim(370, 930)
    ax.set_ylim(0, 1.3)

    colors = cm.get_cmap('Dark2', 9)


    main_name = "brnn_learnrealizable-allparams-longer"#"rnn-wodel_102units_smnist_linebyline_repeat"#brnn-initwithburst_256units_smnist_linebyline_repeat"#"rnn-wodelay_45units_smnist_linebyline"#"brnn200_noncued_moreascs_diffinit"#"brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_wtonly_agn_nodivstart"#lng_lngersim_uniformoffset_furthertrain"

    base_name = "figures_wkof_072521/" + main_name
    base_name_save = "traininfo_wkof_072521/" + main_name
    base_name_model = "models_wkof_072521/" + main_name
    base_name_results = "results_wkof_080121/" + main_name

    training_info = {"losses": [],
        "weights": [],
        "threshes": [],
        "v_resets": [],
        "k_ms": [],
        "k_syns": [],
        "asc_amps": [],
        "asc_rs": [],
        "asc_ks": []
        }

    initburst = False

    dt = 0.05
    sim_time = 10
    nsteps = int(sim_time / dt)

    hid_size = 1
    input_size = 1
    output_size = 1

    targets = torch.empty((1, nsteps, output_size))
    inputs = 0.01 * torch.ones((1, nsteps, input_size))

    train_params = ["thresh", "k_m", "asc_amp", "asc_r", "asc_k"]#, "k_m"]

    target_model = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, dt=dt, output_weight=False)
    target_model.neuron_layer.weight_iv.data = (1 / hid_size) * torch.ones((target_model.neuron_layer.input_size, target_model.neuron_layer.hidden_size))
    # target_model.neuron_layer.thresh.data -= 2
    # target_model.load_state_dict(torch.load("saved_models/" + base_name_model + "_target.pt"))
    learning_model = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, dt=dt, output_weight=False)
    target_model.neuron_layer.asc_amp.data *= 5
    target_model.neuron_layer.trans_asc_r.data *= 2

    with torch.no_grad():
        asc_r1, asc_r2 = 1 - 1e-10, -(1 - 1e-10)
        asc_amp1, asc_amp2 = -1, 1
        target_model.neuron_layer.trans_asc_r[0,:,:] = math.log((1 - asc_r1) / (1 + asc_r1))
        target_model.neuron_layer.trans_asc_r[1,:,:] = math.log((1 - asc_r2) / (1 + asc_r2))

        target_model.neuron_layer.asc_amp[0,:,:] = asc_amp1
        target_model.neuron_layer.asc_amp[1,:,:] = asc_amp2
    # target_model.neuron_layer.thresh.data *= -10

    with torch.no_grad():
        learning_model.neuron_layer.thresh.data = target_model.neuron_layer.thresh.data
        learning_model.neuron_layer.trans_k_m.data = target_model.neuron_layer.trans_k_m.data
        learning_model.neuron_layer.asc_amp.data = target_model.neuron_layer.asc_amp.data
        learning_model.neuron_layer.trans_asc_r.data = target_model.neuron_layer.trans_asc_r.data
        learning_model.neuron_layer.trans_asc_k.data = target_model.neuron_layer.trans_asc_k.data
        learning_model.neuron_layer.weight_iv.data = target_model.neuron_layer.weight_iv.data

        if "asc_amp" in train_params:
            learning_model.neuron_layer.asc_amp.data = 0 * torch.randn((2, 1, hid_size), dtype=torch.float)
        if "asc_r" in train_params:
            learning_model.neuron_layer.trans_asc_r.data = torch.randn((2, 1, hid_size), dtype=torch.float)
        if "asc_k" in train_params:
            learning_model.neuron_layer.trans_asc_k.data = torch.randn((2, 1, hid_size), dtype=torch.float)
        if "k_m" in train_params:
            new_km = 1
            learning_model.neuron_layer.trans_k_m.data = math.log(new_km * dt / (1 - (new_km * dt))) * torch.ones((1, hid_size), dtype=torch.float)
        if "thresh" in train_params:
            learning_model.neuron_layer.thresh.data = torch.randn((1, hid_size), dtype=torch.float)

        # learning_model.load_state_dict(torch.load("saved_models/" + base_name_model + "_learned.pt"))

    target_model.eval()
    # with torch.no_grad():
    #     target_model.reset_state(1)
    #     outputs = target_model(inputs)
    #     targets[0,:,:] = outputs[0, -nsteps:, :]
    # plt.plot(np.arange(nsteps) * dt, targets[0,:,0].detach().numpy(), linewidth=2, color=colors(1), label='target')
    
    # with torch.no_grad():
    #     learning_model.reset_state(1)
    #     outputs = learning_model(inputs)
    #     outputs[0,:,:] = outputs[0, -nsteps:, :]
    #     plt.plot(np.arange(nsteps) * dt, outputs[0,:,0].detach().numpy(), linewidth=2, color=colors(2), label='initial')
    
    # plt.legend()
    # plt.show()
    with torch.no_grad():
        target_model.reset_state(1)
        outputs = target_model(inputs)
        targets[0,:,:] = outputs[0, -nsteps:, :]
    ax.plot(np.arange(nsteps) * dt, torch.mean(targets[0,:,:],-1).detach().numpy(), linewidth=2, color=colors(1), label='target')
    
    with torch.no_grad():
        learning_model.reset_state(1)
        outputs = learning_model(inputs)
        outputs[0,:,:] = outputs[0, -nsteps:, :]
        ax.plot(np.arange(nsteps) * dt, torch.mean(outputs[0,:,:],-1).detach().numpy(), linewidth=2, color=colors(2), label='initial')
        np.savetxt("results/" + base_name_results + "-" + "initialoutputs.csv", np.stack(outputs).reshape((-1, 1)), delimiter=',')

    # Train model
    num_epochs = 5000
    lr = 0.003# 0.01 for thresh, 0.1 for asck

    train_params_real = []
    if "thresh" in train_params:
        train_params_real.append(learning_model.neuron_layer.thresh)
    if "k_m" in train_params:
        train_params_real.append(learning_model.neuron_layer.trans_k_m)
    if "asc_amp" in train_params:
        train_params_real.append(learning_model.neuron_layer.asc_amp)
    if "asc_k" in train_params:
        train_params_real.append(learning_model.neuron_layer.trans_asc_k)
    if "asc_r" in train_params:
        train_params_real.append(learning_model.neuron_layer.trans_asc_r)

    optimizer = torch.optim.Adam(train_params_real, lr=lr)
    losses = []
    loss_fn = nn.MSELoss()
    learning_model.train()
    for epoch in range(num_epochs):
        loss = 0.0
        learning_model.reset_state(1)
        optimizer.zero_grad()

        outputs = learning_model(inputs)
        loss = loss + loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch}/{num_epochs}: loss of {loss.item()}")
        losses.append(loss.item())

        with torch.no_grad():
            print(learning_model.neuron_layer.thresh.grad)
            training_info["k_ms"].append([learning_model.neuron_layer.transform_to_k(learning_model.neuron_layer.trans_k_m[0,j]).item() - learning_model.neuron_layer.transform_to_k(target_model.neuron_layer.trans_k_m[0,j]).item()  + 0.0 for j in range(learning_model.hid_size)])
            training_info["threshes"].append([learning_model.neuron_layer.thresh[0,j].item() - target_model.neuron_layer.thresh[0,j].item()  + 0.0 for j in range(learning_model.hid_size)])
            training_info["asc_ks"].append([learning_model.neuron_layer.transform_to_k(learning_model.neuron_layer.trans_asc_k[j,0,m]).item() - learning_model.neuron_layer.transform_to_k(target_model.neuron_layer.trans_asc_k[j,0,m]).item()  + 0.0 for j in range(learning_model.neuron_layer.num_ascs) for m in range(learning_model.hid_size)])
            training_info["asc_amps"].append([learning_model.neuron_layer.asc_amp[j,0,m].item() - target_model.neuron_layer.asc_amp[j,0,m].item()  + 0.0 for j in range(learning_model.neuron_layer.num_ascs) for m in range(learning_model.hid_size)])
            training_info["asc_rs"].append([learning_model.neuron_layer.transform_to_asc_r(learning_model.neuron_layer.trans_asc_r)[j,0,m].item() - learning_model.neuron_layer.transform_to_asc_r(target_model.neuron_layer.trans_asc_r)[j,0,m].item()  + 0.0 for j in range(learning_model.neuron_layer.num_ascs) for m in range(learning_model.hid_size)])
            training_info["losses"].append(loss.item())

    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "kmoverlearning.csv", np.array(training_info["k_ms"]), delimiter=",")
    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "threshoverlearning.csv", np.array(training_info["threshes"]), delimiter=",")
    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "asckoverlearning.csv", np.array(training_info["asc_ks"]), delimiter=",")
    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "ascroverlearning.csv", np.array(training_info["asc_rs"]), delimiter=",")
    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "ascampoverlearning.csv", np.array(training_info["asc_amps"]), delimiter=",")

    with torch.no_grad():
        learning_model.reset_state(1)
        outputs = learning_model(inputs)

        outputs[0,:,:] = outputs[0, -nsteps:, :]
        ax.plot(np.arange(nsteps) * dt, torch.mean(outputs[0,:,:],-1).detach().numpy(), linewidth=2, color=colors(3), label='learned')

        np.savetxt("results/" + base_name_results + "-" + "finaloutputs.csv", np.stack(outputs).reshape((-1, 1)), delimiter=',')
        np.savetxt("results/" + base_name_results + "-" + "targets.csv", targets.reshape((-1,1)), delimiter=',')
    ax.set_xlabel('time (ms)', labelpad=10)
    ax.set_ylabel('firing probability', labelpad=10)
    ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    plt.savefig("figures/" + base_name + "_outputs", dpi=300, transparent=False, bbox_inches='tight')
    plt.close()

    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "losses.csv", np.array(losses), delimiter=",")

    membrane_parameters = np.zeros((hid_size, 2))
    membrane_parameters[:, 0] = learning_model.neuron_layer.thresh.detach().numpy().reshape(-1) - target_model.neuron_layer.thresh.detach().numpy().reshape(-1)
    membrane_parameters[:, 1] = learning_model.neuron_layer.transform_to_k(learning_model.neuron_layer.trans_k_m).detach().numpy().reshape(-1) - target_model.neuron_layer.transform_to_k(target_model.neuron_layer.trans_k_m).detach().numpy().reshape(-1)
    np.savetxt("results/" + base_name_results + "-" + "membraneparams.csv", membrane_parameters, delimiter=',')

    asc_parameters = np.zeros((hid_size * 2, 3))
    asc_parameters[:, 0] = learning_model.neuron_layer.transform_to_k(learning_model.neuron_layer.trans_asc_k)[:,0,:].detach().numpy().reshape(-1) - target_model.neuron_layer.transform_to_k(target_model.neuron_layer.trans_asc_k)[:,0,:].detach().numpy().reshape(-1)
    asc_parameters[:, 1] = learning_model.neuron_layer.transform_to_asc_r(learning_model.neuron_layer.trans_asc_r)[:,0,:].detach().numpy().reshape(-1) - target_model.neuron_layer.transform_to_asc_r(target_model.neuron_layer.trans_asc_r)[:,0,:].detach().numpy().reshape(-1)
    asc_parameters[:, 2] = learning_model.neuron_layer.asc_amp[:,0,:].detach().numpy().reshape(-1) - target_model.neuron_layer.asc_amp[:,0,:].detach().numpy().reshape(-1)
    np.savetxt("results/" + base_name_results + "-ascparams.csv", asc_parameters, delimiter=',')

    torch.save(learning_model.state_dict(), "saved_models/" + base_name_model + "_learned.pt")
    torch.save(target_model.state_dict(), "saved_models/" + base_name_model + "_target.pt")
    
    # colors = ["sienna", "darkorange", "purple", "slateblue", "aqua", "springgreen", "fuchsia", "plum", "darkorchid", "mediumblue", "cornflowerblue", "skyblue", "aquamarine", "springgreen", "green", "lightgreen"]
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    # ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
    # ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    # ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.plot(training_info["losses"], linewidth=2, color=colors(1))
    ax.set_xlabel('epoch', labelpad=10)
    ax.set_ylabel('MSE', labelpad=10)
    plt.savefig("figures/" + base_name + "_losses", dpi=300, transparent=False, bbox_inches='tight')
    plt.close()
    torch.save(training_info["losses"], "traininfo/" + base_name_save + "_losses.pt")
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    # ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
    # ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    # ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    min_ylim = -1
    max_ylim = 1
    i = 0
    for name in ["threshes", "k_ms", "asc_amps", "asc_rs", "asc_ks"]:#, "k_ms", "asc_amps", "asc_rs", "asc_ks"]:
        print(name)
        i += 1
        _, l = np.array(training_info[name]).shape
        for j in range(l):
            ax.plot(np.array(training_info[name])[:, j], alpha = 0.5, label = name if j == 0 else "", linewidth=2, color=colors(i))
            min_ylim = min(min_ylim, min(np.array(training_info[name])[:,j]))
            max_ylim = max(max_ylim, max(np.array(training_info[name])[:,j]))
    ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    ax.axhline(color='k')
    ax.set_ylim(min_ylim, max_ylim)
    ax.set_xlabel('epoch', labelpad=10)
    ax.set_ylabel('parameter difference', labelpad=10)
    plt.savefig("figures/" + base_name + "_parameter-diffs", dpi=300, transparent=False, bbox_inches='tight')
    plt.close()

    with open("traininfo/" + base_name_save + ".pickle", 'wb') as handle:
                pickle.dump(training_info, handle)


def main2():
    main_name = "brnn_learnrealizable-losses"#"rnn-wodel_102units_smnist_linebyline_repeat"#brnn-initwithburst_256units_smnist_linebyline_repeat"#"rnn-wodelay_45units_smnist_linebyline"#"brnn200_noncued_moreascs_diffinit"#"brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_wtonly_agn_nodivstart"#lng_lngersim_uniformoffset_furthertrain"

    # base_name = "figures_wkof_072521/" + main_name
    # base_name_save = "traininfo_wkof_072521/" + main_name
    # base_name_model = "models_wkof_072521/" + main_name
    base_name_results = "results_wkof_080121/" + main_name
    
    dt = 0.05
    sim_time = 10
    nsteps = int(sim_time / dt)

    hid_size = 1
    input_size = 1
    output_size = 1

    targets = torch.empty((1, nsteps, output_size))
    inputs = torch.ones((1, nsteps, input_size))

    train_params = ["thresh", "k_m"]

    target_model = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, dt=dt, output_weight=False)
    target_model.neuron_layer.asc_amp.data *= 2
    target_model.neuron_layer.trans_asc_r.data *= 2

    target_model.eval()
    target_model.reset_state(1)
    targets = target_model(inputs)
    loss_fn = nn.MSELoss()
    
    # Study thresh
    learning_model = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, dt=dt, output_weight=False)
    with torch.no_grad():
        learning_model.neuron_layer.thresh.data = target_model.neuron_layer.thresh.data
        learning_model.neuron_layer.trans_k_m.data = target_model.neuron_layer.trans_k_m.data
        learning_model.neuron_layer.asc_amp.data = target_model.neuron_layer.asc_amp.data
        learning_model.neuron_layer.trans_asc_r.data = target_model.neuron_layer.trans_asc_r.data
        learning_model.neuron_layer.trans_asc_k.data = target_model.neuron_layer.trans_asc_k.data
        learning_model.neuron_layer.weight_iv.data = target_model.neuron_layer.weight_iv.data
    torch.save(learning_model.state_dict(), "random.pt")

    learning_model.load_state_dict(torch.load("random.pt"))
    threshes = np.arange(-20, 20, 2)
    thresh_results = np.zeros((len(threshes), 2))
    for i in range(len(threshes)):
        t = threshes[i]
        learning_model.reset_state(1)
        with torch.no_grad():
            learning_model.neuron_layer.thresh.data[0,0] = t
        outputs = learning_model.forward(inputs)
        thresh_results[i, 0] = t
        thresh_results[i,1] = loss_fn(outputs, targets).item()
    np.savetxt("results/" + base_name_results + "-" + "threshes.csv", thresh_results, delimiter=',')

    learning_model.load_state_dict(torch.load("random.pt"))
    k_ms = np.arange(1e-10, 5, 0.01)
    k_m_results = np.zeros((len(k_ms), 2))
    for i in range(len(k_ms)):
        km = k_ms[i]
        trans_k_m = math.log(km * dt / (1 - (km * dt)))
        learning_model.reset_state(1)
        with torch.no_grad():
            learning_model.neuron_layer.trans_k_m.data[0,0] = trans_k_m
        outputs = learning_model.forward(inputs)
        k_m_results[i, 0] = km
        k_m_results[i,1] = loss_fn(outputs, targets).item()
    np.savetxt("results/" + base_name_results + "-" + "kms.csv", k_m_results, delimiter=',')

    learning_model.load_state_dict(torch.load("random.pt"))
    asc_k_1s = np.arange(1e-10, 5, 0.01)
    asc_k_1_results = np.zeros((len(asc_k_1s), 2))
    for i in range(len(asc_k_1s)):
        asc_k1 = asc_k_1s[i]
        trans_asc_k1 = math.log(asc_k1 * dt / (1 - (asc_k1 * dt)))
        learning_model.reset_state(1)
        with torch.no_grad():
            learning_model.neuron_layer.trans_asc_k.data[0,0,0] = trans_asc_k1
        outputs = learning_model.forward(inputs)
        asc_k_1_results[i, 0] = asc_k1
        asc_k_1_results[i,1] = loss_fn(outputs, targets).item()
    np.savetxt("results/" + base_name_results + "-" + "asck.csv", asc_k_1_results, delimiter=',')

    learning_model.load_state_dict(torch.load("random.pt"))
    asc_r_1s = np.arange(-1, 1, 0.01)
    asc_r1_results = np.zeros((len(asc_r_1s), 2))
    for i in range(len(asc_r_1s)):
        asc_r1 = asc_r_1s[i]
        trans_asc_r1 = math.log((1 - asc_r1) / (1 + asc_r1))
        learning_model.reset_state(1)
        with torch.no_grad():
            learning_model.neuron_layer.trans_asc_r.data[1,0,0] = 0
            learning_model.neuron_layer.trans_asc_r.data[0,0,0] = trans_asc_r1
        outputs = learning_model.forward(inputs)
        asc_r1_results[i, 0] = asc_r1
        asc_r1_results[i,1] = loss_fn(outputs, targets).item()
    np.savetxt("results/" + base_name_results + "-" + "ascr.csv", asc_r1_results, delimiter=',')

    learning_model.load_state_dict(torch.load("random.pt"))
    asc_amp_1s = np.arange(-100, 100, 10)
    asc_amp_1_results = np.zeros((len(asc_amp_1s), 2))
    for i in range(len(asc_amp_1s)):
        asc_amp_1 = asc_amp_1s[i]
        learning_model.reset_state(1)
        with torch.no_grad():
            learning_model.neuron_layer.asc_amp.data[1,0,0] = 0
            learning_model.neuron_layer.asc_amp.data[0,0,0] = asc_amp_1
        outputs = learning_model.forward(inputs)
        asc_amp_1_results[i, 0] = asc_amp_1
        asc_amp_1_results[i,1] = loss_fn(outputs, targets).item()
    np.savetxt("results/" + base_name_results + "-" + "ascamp.csv", asc_amp_1_results, delimiter=',')

if __name__ == '__main__':
        main()
