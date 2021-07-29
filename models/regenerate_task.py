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
    main_name = "brnn_learnrealizable"#"rnn-wodel_102units_smnist_linebyline_repeat"#brnn-initwithburst_256units_smnist_linebyline_repeat"#"rnn-wodelay_45units_smnist_linebyline"#"brnn200_noncued_moreascs_diffinit"#"brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_wtonly_agn_nodivstart"#lng_lngersim_uniformoffset_furthertrain"

    base_name = "figures_wkof_071821/" + main_name
    base_name_save = "traininfo_wkof_071821/" + main_name
    base_name_model = "models_wkof_071821/" + main_name

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
    sim_time = 5
    nsteps = int(sim_time / dt)

    hid_size = 1
    input_size = 1
    output_size = 1

    targets = torch.empty((1, nsteps, output_size))
    inputs =torch.ones((1, nsteps, input_size))

    target_model = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, dt=dt, initburst=True, output_weight=False)
    # target_model.load_state_dict(torch.load("saved_models/" + base_name_model + "_target.pt"))
    learning_model = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, dt=dt, initburst=False, output_weight=False)
    with torch.no_grad():
        learning_model.neuron_layer.thresh *= random.uniform(-5,5)
        learning_model.neuron_layer.ln_k_m *= random.uniform(-2,2) / math.log(.05)
        learning_model.neuron_layer.asc_amp.data = target_model.neuron_layer.asc_amp.data
        learning_model.neuron_layer.asc_r.data = target_model.neuron_layer.asc_r.data
        learning_model.neuron_layer.ln_asc_k.data = target_model.neuron_layer.ln_asc_k.data

        # learning_model.load_state_dict(torch.load("saved_models/" + base_name_model + "_learned.pt"))

    target_model.eval()
    with torch.no_grad():
        target_model.reset_state(1)
        outputs = target_model(inputs)
        targets[0,:,:] = outputs[0, -nsteps:, :]
    plt.plot(np.arange(nsteps) * dt, targets[0,:,0].detach().numpy(), label="target")
    
    with torch.no_grad():
        learning_model.reset_state(1)
        outputs = learning_model(inputs)
        outputs[0,:,:] = outputs[0, -nsteps:, :]
        plt.plot(np.arange(nsteps) * dt, outputs[0,:,0].detach().numpy(), label="initial")
    # plt.legend()
    # plt.show()
    # Train model
    num_epochs = 500
    lr = 0.05

    optimizer = torch.optim.Adam(learning_model.parameters(), lr=lr)
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

        training_info["k_ms"].append([torch.exp(learning_model.neuron_layer.ln_k_m[0,j]) - torch.exp(target_model.neuron_layer.ln_k_m[0,j])  + 0.0 for j in range(learning_model.hid_size)])
        training_info["threshes"].append([learning_model.neuron_layer.thresh[0,j] - target_model.neuron_layer.thresh[0,j]  + 0.0 for j in range(learning_model.hid_size)])
        training_info["asc_ks"].append([torch.exp(learning_model.neuron_layer.ln_asc_k[j,0,m]) - torch.exp(target_model.neuron_layer.ln_asc_k[j,0,m])  + 0.0 for j in range(learning_model.neuron_layer.num_ascs) for m in range(learning_model.hid_size)])
        training_info["asc_amps"].append([learning_model.neuron_layer.asc_amp[j,0,m] - target_model.neuron_layer.asc_amp[j,0,m]  + 0.0 for j in range(learning_model.neuron_layer.num_ascs) for m in range(learning_model.hid_size)])
        training_info["asc_rs"].append([learning_model.neuron_layer.asc_r[j,0,m] - target_model.neuron_layer.asc_r[j,0,m]  + 0.0 for j in range(learning_model.neuron_layer.num_ascs) for m in range(learning_model.hid_size)])
        training_info["losses"].append(loss.item())

    with torch.no_grad():
        learning_model.reset_state(1)
        outputs = learning_model(inputs)
        outputs[0,:,:] = outputs[0, -nsteps:, :]
        plt.plot(np.arange(nsteps) * dt, outputs[0,:,0].detach().numpy(), label="learned")
    plt.legend()
    plt.savefig("figures/" + base_name + "_outputs")
    plt.close()

    torch.save(learning_model.state_dict(), "saved_models/" + base_name_model + "_learned.pt")
    torch.save(target_model.state_dict(), "saved_models/" + base_name_model + "_target.pt")
    
    colors = ["sienna", "darkorange", "purple", "slateblue", "aqua", "springgreen", "fuchsia", "plum", "darkorchid", "mediumblue", "cornflowerblue", "skyblue", "aquamarine", "springgreen", "green", "lightgreen"]
    
    plt.plot(training_info["losses"])
    plt.xlabel("epoch #")
    plt.ylabel("loss")
    plt.savefig("figures/" + base_name + "_losses")
    plt.close()
    torch.save(training_info["losses"], "traininfo/" + base_name_save + "_losses.pt")

    i = 0
    for name in ["threshes", "k_ms", "asc_amps", "asc_rs", "asc_ks"]:
        print(name)
        i += 1
        _, l = np.array(training_info[name]).shape
        for j in range(l):
                plt.plot(np.array(training_info[name])[:, j], color = colors[i], alpha = 0.5, label = name if j == 0 else "")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('parameter difference')
    plt.savefig("figures/" + base_name + "_parameter-diffs")
    plt.close()
        
    with open("traininfo/" + base_name_save + ".pickle", 'wb') as handle:
                pickle.dump(training_info, handle)

if __name__ == '__main__':
        main()
