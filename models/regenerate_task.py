import matplotlib as mpl
mpl.use('Agg')

# author: @chloewin
# 03/07/21
import pickle
import datetime
import utils as ut
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

    font = {'family' : 'arial',
        # 'weight' : 'bold',
        'size'   : 22}

    mpl.rc('font', **font)

    colors = cm.get_cmap('Set2', 5)


    main_name = "brnn_learnrealizable"#"rnn-wodel_102units_smnist_linebyline_repeat"#brnn-initwithburst_256units_smnist_linebyline_repeat"#"rnn-wodelay_45units_smnist_linebyline"#"brnn200_noncued_moreascs_diffinit"#"brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_wtonly_agn_nodivstart"#lng_lngersim_uniformoffset_furthertrain"

    base_name = "figures_wkof_072521/" + main_name
    base_name_save = "traininfo_wkof_072521/" + main_name
    base_name_model = "models_wkof_072521/" + main_name

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
    sim_time = 4
    nsteps = int(sim_time / dt)

    hid_size = 1
    input_size = 1
    output_size = 1

    targets = torch.empty((1, nsteps, output_size))
    inputs = torch.ones((1, nsteps, input_size))

    target_model = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, dt=dt, output_weight=False)
    # target_model.load_state_dict(torch.load("saved_models/" + base_name_model + "_target.pt"))
    learning_model = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, dt=dt, output_weight=False)
    with torch.no_grad():
        learning_model.neuron_layer.thresh.data = target_model.neuron_layer.thresh.data
        learning_model.neuron_layer.thresh.data = torch.randn((1, hid_size), dtype=torch.float)
        learning_model.neuron_layer.trans_k_m.data = target_model.neuron_layer.trans_k_m.data
        # learning_model.neuron_layer.trans_k_m.data *= random.uniform(-1,1)
        learning_model.neuron_layer.asc_amp.data = target_model.neuron_layer.asc_amp.data
        learning_model.neuron_layer.trans_asc_r.data = target_model.neuron_layer.trans_asc_r.data
        learning_model.neuron_layer.trans_asc_k.data = target_model.neuron_layer.trans_asc_k.data
        learning_model.neuron_layer.weight_iv.data = target_model.neuron_layer.weight_iv.data

        # learning_model.load_state_dict(torch.load("saved_models/" + base_name_model + "_learned.pt"))

    target_model.eval()
    # with torch.no_grad():
    #     target_model.reset_state(1)
    #     outputs = target_model(inputs)
    #     targets[0,:,:] = outputs[0, -nsteps:, :]
    # ax.plot(np.arange(nsteps) * dt, targets[0,:,0].detach().numpy(), linewidth=2, color=colors(1), label='target')
    
    # with torch.no_grad():
    #     learning_model.reset_state(1)
    #     outputs = learning_model(inputs)
    #     outputs[0,:,:] = outputs[0, -nsteps:, :]
    #     ax.plot(np.arange(nsteps) * dt, outputs[0,:,0].detach().numpy(), linewidth=2, color=colors(1), label='target')
    
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

    # Train model
    num_epochs = 2000
    lr = 0.001

    optimizer = torch.optim.Adam([learning_model.neuron_layer.thresh], lr=lr)
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

        with torch.no_grad():
            print(learning_model.neuron_layer.thresh.grad)
            training_info["k_ms"].append([learning_model.neuron_layer.transform_to_k(learning_model.neuron_layer.trans_k_m[0,j]) - learning_model.neuron_layer.transform_to_k(target_model.neuron_layer.trans_k_m[0,j])  + 0.0 for j in range(learning_model.hid_size)])
            training_info["threshes"].append([learning_model.neuron_layer.thresh[0,j] - target_model.neuron_layer.thresh[0,j]  + 0.0 for j in range(learning_model.hid_size)])
            training_info["asc_ks"].append([learning_model.neuron_layer.transform_to_k(learning_model.neuron_layer.trans_asc_k[j,0,m]) - learning_model.neuron_layer.transform_to_k(target_model.neuron_layer.trans_asc_k[j,0,m])  + 0.0 for j in range(learning_model.neuron_layer.num_ascs) for m in range(learning_model.hid_size)])
            training_info["asc_amps"].append([learning_model.neuron_layer.asc_amp[j,0,m] - target_model.neuron_layer.asc_amp[j,0,m]  + 0.0 for j in range(learning_model.neuron_layer.num_ascs) for m in range(learning_model.hid_size)])
            training_info["asc_rs"].append([learning_model.neuron_layer.transform_to_asc_r(learning_model.neuron_layer.trans_asc_r)[j,0,m] - learning_model.neuron_layer.transform_to_asc_r(target_model.neuron_layer.trans_asc_r)[j,0,m]  + 0.0 for j in range(learning_model.neuron_layer.num_ascs) for m in range(learning_model.hid_size)])
            training_info["losses"].append(loss.item())

    with torch.no_grad():
        learning_model.reset_state(1)
        outputs = learning_model(inputs)
        outputs[0,:,:] = outputs[0, -nsteps:, :]
        ax.plot(np.arange(nsteps) * dt, torch.mean(outputs[0,:,:],-1).detach().numpy(), linewidth=2, color=colors(3), label='learned')
    ax.set_xlabel('time (ms)', labelpad=10)
    ax.set_ylabel('firing probability', labelpad=10)
    ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    plt.savefig("figures/" + base_name + "_outputs", dpi=300, transparent=False, bbox_inches='tight')
    plt.close()

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

    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))

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

    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    i = 0
    for name in ["threshes"]:#, "k_ms", "asc_amps", "asc_rs", "asc_ks"]:
        print(name)
        i += 1
        _, l = np.array(training_info[name]).shape
        for j in range(l):
                ax.plot(np.array(training_info[name])[:, j], alpha = 0.5, label = name if j == 0 else "", linewidth=2, color=colors(i))
    ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    ax.set_xlabel('epoch', labelpad=10)
    ax.set_ylabel('parameter difference', labelpad=10)
    plt.savefig("figures/" + base_name + "_parameter-diffs", dpi=300, transparent=False, bbox_inches='tight')
    plt.close()

    with open("traininfo/" + base_name_save + ".pickle", 'wb') as handle:
                pickle.dump(training_info, handle)

if __name__ == '__main__':
        main()
