"""
This file is used for miscellaneous plotting and analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.utils.data as tud

import utils as ut
from networks import RNNFC, BNNFC

fontsize = 18

folder_loss = "traininfo_wkof_051621/"
losses_rnn = torch.load("traininfo/" + folder_loss + "rnn200_sussillo8_batched_hisgmav_predrive_scaleasc_losses.pt")
losses_glif = torch.load("traininfo/" + folder_loss + "brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_agn_nodivstart_losses.pt")
losses_glifwt = torch.load("traininfo/" + folder_loss + "brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_wtonly_agn_nodivstart_losses.pt")
# losses_glifpar = torch.load("traininfo/" + folder_loss + "brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_paramonly_losses.pt")
losses_noasc = torch.load("traininfo/" + folder_loss + "brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_noasc_losses.pt")

# plt.plot(losses_rnn, label = "RNN-weights")
plt.plot(losses_glif, 'purple', label = "GLIF_ASC-both")
plt.plot(losses_glifwt, 'orange', label = "GLIF_ASC-weights")
# plt.plot(losses_glifpar, label = "GLIF_ASC-parameters")
plt.plot(losses_noasc, 'green', label = "LIF-both")

plt.legend(fontsize = fontsize - 2)
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.xlim([0,250]) # TODO: please change as needed
plt.xlabel("epoch #", fontsize = fontsize)
plt.ylabel("mse loss", fontsize = fontsize)
plt.show()

def plot_overall_response(model):
    sim_time = 10
    dt = 0.05
    nsteps = int(sim_time / dt)

    num_freqs = 8
    freq_min = 0.001
    freq_max = 0.6
    
    freqs = 10 ** np.linspace(np.log10(freq_min), np.log10(freq_max), num=num_freqs)

    inputs, targets = ut.create_sines_cued(sim_time, dt, amp = 1, noise_mean = 0, noise_std = 0, freqs = freqs, input_size = 16)
    traindataset = ut.create_dataset(inputs, targets, input_size)
    trainloader = tud.DataLoader(traindataset, batch_size = 1)

    for batch_ndx, sample in enumerate(trainloader):
        input, target = sample
        model.reset_state(1)
        with torch.no_grad():
            model(input)
        output = model(input)
        plt.plot(np.arange(len(output[0])) * dt, output[0,:,0].detach().numpy(), 'k', label=f"predicted")
        plt.plot(np.arange(len(output[0])) * dt, target[0,:,0], 'k--', label = "target")
        plt.xlabel('time (ms)', fontsize = fontsize)
        plt.ylabel('firing', fontsize = fontsize)
        plt.legend(fontsize = fontsize - 2)
        plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.show()

# Neuronal Response Curves
def plot_responses(model):
    sim_time = 20
    dt = 0.05
    nsteps = int(sim_time / dt)
    input = torch.ones(1, nsteps, hid_size + input_size)
    outputs = torch.zeros(1, nsteps, hid_size)

    firing = torch.zeros((1, hid_size))
    voltage = torch.zeros((1, hid_size))
    syncurrent = torch.zeros((1, hid_size))
    ascurrents = torch.zeros((2, 1, hid_size))

   

    for step in range(nsteps):
        x = input[:, step, :]
        firing, voltage, ascurrents, syncurrent = model.neuron_layer(x, firing, voltage, ascurrents, syncurrent)
        outputs[:, step, :] = firing
    
    for neuron_idx in range(hid_size):
        if random.random() < 0.05:
            print()
            print(torch.exp(model.neuron_layer.ln_k_m[0,neuron_idx]))
            print(model.neuron_layer.thresh[0,neuron_idx])
            print(torch.exp(model.neuron_layer.ln_asc_k[:,0,neuron_idx]))
            print(model.neuron_layer.asc_r[:,0,neuron_idx])
            print(model.neuron_layer.asc_amp[:,0,neuron_idx])

            plt.plot(np.arange(0, sim_time, step = dt), outputs.detach().numpy()[0, :, neuron_idx], 'k')
            plt.xlabel('time (ms)', fontsize = fontsize)
            plt.ylabel('firing rate', fontsize = fontsize)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)
            plt.show()

input_size = 16
hid_size = 200
output_size = 1


x = np.arange(-10, 10, step = 0.1)
y = torch.tanh(torch.from_numpy(x).float())
plt.plot(x, y.detach().numpy(), 'o')
plt.show()

x = np.arange(-10, 10, step = 0.1)
y = torch.relu(torch.from_numpy(x).float())
plt.plot(x, y.detach().numpy(), 'o')
plt.show()

x = np.arange(-10, 10, step = 0.1)
y = torch.from_numpy(x > 0)#torch.sign(torch.from_numpy(x).float())
plt.plot(x, y.detach().numpy(), 'o')
plt.vlines(0,0,1, linestyles='dashed')
plt.show()

model_glif = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size)
model_glif.load_state_dict(torch.load("saved_models/models_wkof_051621/brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_agn_nodivstart.pt"))

plt.hist(torch.exp(model_glif.neuron_layer.ln_k_m[0,:]).detach().numpy(), color = 'k')
plt.xlabel('k_m', fontsize = fontsize)
plt.ylabel('counts', fontsize = fontsize)
plt.xlim([0,0.05])
plt.show()

plt.hist(model_glif.neuron_layer.thresh[0,:].detach().numpy(), color = 'k')
plt.xlabel('threshold', fontsize = fontsize)
plt.ylabel('counts', fontsize = fontsize)
plt.xlim([-0.5,0.5])
plt.show()

plt.hist(torch.cat((model_glif.neuron_layer.asc_amp[0, 0,:], model_glif.neuron_layer.asc_amp[1, 0,:]), axis = 0).detach().numpy(), color = 'k', bins = 50)
plt.xlabel('a_j', fontsize = fontsize)
plt.ylabel('counts', fontsize = fontsize)
plt.show()

plt.hist(torch.cat((model_glif.neuron_layer.asc_r[0, 0,:], model_glif.neuron_layer.asc_amp[1, 0,:]), axis = 0).detach().numpy(), color = 'k', bins = 50)
plt.xlabel('r_j', fontsize = fontsize)
plt.ylabel('counts', fontsize = fontsize)
plt.show()

plt.hist(torch.cat((torch.exp(model_glif.neuron_layer.ln_asc_k[0, 0,:]), torch.exp(model_glif.neuron_layer.ln_asc_k[1, 0,:])), axis = 0).detach().numpy(), color = 'k', bins = 50)
plt.xlabel('k_j', fontsize = fontsize)
plt.ylabel('counts', fontsize = fontsize)
plt.show()

# print(torch.mean(model_glif.neuron_layer.weight_iv))
plot_overall_response(model_glif)
nn.init.constant_(model_glif.neuron_layer.weight_iv, 0.0001)
plot_responses(model_glif)