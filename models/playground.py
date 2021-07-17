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

# folder_loss = "traininfo_wkof_053021/"
# losses_rnn = torch.load("traininfo/" + folder_loss + "5dsine_rrnn_short060621_10ms_spontaneous_losses.pt")
# losses_glif = torch.load("traininfo/" + folder_loss + "5dsine_brnn_short060621_10ms_spontaneous_losses.pt")
# plt.plot(losses_rnn, color = 'orange', label = "RNN")
# plt.plot(losses_glif, 'purple', label = "GLIF_ASC")
# losses_rnn = torch.load("traininfo/" + folder_loss + "rnn200_sussillo8_batched_hisgmav_predrive_scaleasc_losses.pt")
# losses_glif = torch.load("traininfo/" + folder_loss + "brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_agn_nodivstart_losses.pt")
# losses_glifwt = torch.load("traininfo/" + folder_loss + "brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_wtonly_agn_nodivstart_losses.pt")
# # losses_glifpar = torch.load("traininfo/" + folder_loss + "brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_paramonly_losses.pt")
# losses_noasc = torch.load("traininfo/" + folder_loss + "brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_noasc_losses.pt")

# # plt.plot(losses_rnn, label = "RNN-weights")
# plt.plot(losses_glif, 'purple', label = "GLIF_ASC-both")
# plt.plot(losses_glifwt, 'orange', label = "GLIF_ASC-weights")
# # plt.plot(losses_glifpar, label = "GLIF_ASC-parameters")
# plt.plot(losses_noasc, 'green', label = "LIF-both")

# plt.legend(fontsize = fontsize - 2)
# plt.xticks(fontsize = fontsize)
# plt.yticks(fontsize = fontsize)
# plt.xlim([0,250]) # TODO: please change as needed
# plt.xlabel("epoch #", fontsize = fontsize)
# plt.ylabel("mse loss", fontsize = fontsize)
# plt.show()

def plot_overall_response(model):
    sim_time = 40
    dt = 0.05
    nsteps = int(sim_time / dt)

    num_freqs = 10#8
    freq_min = 0.01#01
    freq_max = 0.6
    
    freqs = 10 ** np.linspace(np.log10(freq_min), np.log10(freq_max), num=num_freqs)

    hid_size = 128#64
    input_size = 20#8
    output_size = 1
    inputs, targets = ut.create_sines_cued(sim_time, dt, amp = 1, noise_mean = 0, noise_std = 0, freqs = freqs, input_size = input_size)
    traindataset = ut.create_dataset(inputs, targets, input_size)

    # inputs, targets = ut.create_multid_pattern(sim_time, dt, 1, 0, 0, freqs, input_size)
    # traindataset = ut.create_dataset(inputs, targets, input_size, output_size)

    trainloader = tud.DataLoader(traindataset, batch_size = 1)

    for batch_ndx, sample in enumerate(trainloader):
        input, target = sample
        model.reset_state(1)
        with torch.no_grad():
            output = model(input)
        # output = model(input)
        for j in range(1):
            plt.plot(np.arange(len(output[0])) * dt, output[0,:,j].detach().numpy(), 'k', label=f"predicted")
            plt.plot(np.arange(len(output[0])) * dt, target[0,:,j], 'k--', label = "target")
            plt.xlabel('time (ms)', fontsize = fontsize)
            plt.ylabel('firing', fontsize = fontsize)
            plt.legend(fontsize = fontsize - 2)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)
            plt.show()

# Neuronal Response Curves
def plot_responses(model):
    sim_time = 1000
    dt = 0.05
    nsteps = int(sim_time / dt)
    input = -5.5 * 0.0001 * torch.ones(1, nsteps, output_size + input_size)
    outputs = torch.zeros(1, nsteps, hid_size)

    firing = torch.zeros((1, hid_size))
    voltage = torch.zeros((1, hid_size))
    syncurrent = torch.zeros((1, hid_size))
    ascurrents = torch.zeros((2, 1, hid_size))

    for step in range(nsteps):
        x = input[:, step, :]
        firing, voltage, ascurrents, syncurrent = model.neuron_layer(x, firing, voltage, ascurrents, syncurrent)
        outputs[:, step, :] = firing
    
    for neuron_idx in [4]:#range(hid_size):
        if random.random() < 1:
            print()
            print(torch.exp(model.neuron_layer.ln_k_m[0,neuron_idx]))
            print(model.neuron_layer.thresh[0,neuron_idx])
            print(torch.exp(model.neuron_layer.ln_asc_k[:,0,neuron_idx]))
            print(model.neuron_layer.asc_r[:,0,neuron_idx])
            print(model.neuron_layer.asc_amp[:,0,neuron_idx])

            plt.plot(np.arange(0, sim_time, step = dt), outputs.detach().numpy()[0, :, neuron_idx])
            plt.xlabel('time (ms)', fontsize = fontsize)
            plt.ylabel('firing rate', fontsize = fontsize)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)
    plt.show()

def plot_ficurve(model):
    # x_ins = np.arange(-100,100,1)

    sim_time = 1000
    dt = 0.05
    nsteps = int(sim_time / dt)

    # i_syns = 28 * x_ins * 0.0001
    i_syns = np.arange(-0.1, 0.1, step=0.001)

    input = torch.zeros(1, nsteps, glif_input_size)
    outputs = torch.zeros(len(i_syns), nsteps, hid_size)

    for i in range(len(i_syns)):
        firing = torch.zeros((input.shape[0], hid_size))
        voltage = torch.zeros((input.shape[0], hid_size))
        syncurrent = torch.zeros((input.shape[0], hid_size))
        ascurrents = torch.zeros((2, input.shape[0], hid_size))
        outputs_temp = torch.zeros(1, nsteps, hid_size)

        model.neuron_layer.I0 = i_syns[i]
        for step in range(nsteps):
            x = input[:, step, :]
            firing, voltage, ascurrents, syncurrent = model.neuron_layer(x, firing, voltage, ascurrents, syncurrent)
            outputs_temp[:, step, :] = firing
        outputs[i,:,:] = outputs_temp[0,:,:]

    f_rates = torch.mean(outputs, dim=1).detach().numpy()
    print(f"f_rates.shape = {f_rates.shape}")

    slopes = np.zeros(hid_size)
    for i in range(hid_size):
        A = np.vstack([i_syns, np.ones_like(i_syns)]).T
        m, c = np.linalg.lstsq(A, f_rates[:,i])[0]
        slopes[i] = m * sim_time / dt
    
    plt.hist(slopes, color = 'k', bins = 50)
    plt.xlabel('f-i curve slope', fontsize = fontsize)
    plt.ylabel('counts', fontsize = fontsize)
    plt.savefig("figures/figures_wkof_071121/f-i-curve-slopes_brnn.png")

input_size = 28
hid_size = 256
output_size = 10

glif_input_size = output_size + input_size

# hid_size = 128#64
# input_size = 20#8
# output_size = 1

model_glif = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size)
# model_glif.load_state_dict(torch.load("saved_models/models_wkof_071121/rnn-wodel_103units_smnist_linebyline.pt"))

model_glif.load_state_dict(torch.load("saved_models/models_wkof_071121/brnn-initwithburst_256units_smnist_linebyline.pt"))
nn.init.constant_(model_glif.neuron_layer.weight_iv, 1)
# print(torch.mean(model_glif.neuron_layer.weight_iv))
plot_ficurve(model_glif)
quit()
# plt.hist(torch.cat((model_glif.neuron_layer.weight_ih.reshape(-1,1), model_glif.neuron_layer.weight_hh.reshape(-1,1), model_glif.output_linear.weight.reshape(-1, 1)), axis = 0).detach().numpy(), color = 'k', bins = 50)
# plt.xlabel('weights', fontsize = fontsize)
# plt.ylabel('counts', fontsize = fontsize)
# plt.show()

with torch.no_grad():
    nn.init.constant_(model_glif.neuron_layer.weight_iv, 1)
    # nn.init.constant_(model_glif.neuron_layer.asc_amp, 10e-5)
    plot_responses(model_glif)

quit()
# with torch.no_grad():
#     plot_overall_response(model_glif)

# input_size = 16
# hid_size = 200
# output_size = 1


# x = np.arange(-10, 10, step = 0.1)
# y = torch.tanh(torch.from_numpy(x).float())
# plt.plot(x, y.detach().numpy(), 'o')
# plt.show()

# x = np.arange(-10, 10, step = 0.1)
# y = torch.relu(torch.from_numpy(x).float())
# plt.plot(x, y.detach().numpy(), 'o')
# plt.show()

# x = np.arange(-10, 10, step = 0.1)
# y = torch.from_numpy(x > 0)#torch.sign(torch.from_numpy(x).float())
# plt.plot(x, y.detach().numpy(), 'o')
# plt.vlines(0,0,1, linestyles='dashed')
# plt.show()

# model_glif = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size)
# model_glif.load_state_dict(torch.load("saved_models/models_wkof_051621/brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_agn_nodivstart.pt"))

# plot_overall_response(model_glif)
# print(torch.exp(model_glif.neuron_layer.ln_k_m).shape)
plt.hist(torch.exp(model_glif.neuron_layer.ln_k_m[0,:]).detach().numpy(), color = 'k', bins=50)
plt.xlabel('k_m (ms)', fontsize = fontsize)
plt.ylabel('counts', fontsize = fontsize)
# plt.xlim([0,0.05])
plt.show()

# plt.hist(torch.exp(model_glif.neuron_layer.ln_k_syn[0,:]).detach().numpy(), color = 'k', bins=50)
# plt.xlabel('k_syn', fontsize = fontsize)
# plt.ylabel('counts', fontsize = fontsize)
# # plt.xlim([0,0.05])
# plt.show()

plt.hist(model_glif.neuron_layer.thresh[0,:].detach().numpy(), color = 'k', bins=50)
plt.xlabel('threshold (mV)', fontsize = fontsize)
plt.ylabel('counts', fontsize = fontsize)
# plt.xlim([-0.5,0.5])
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
plt.xlabel('k_j (ms)', fontsize = fontsize)
plt.ylabel('counts', fontsize = fontsize)
plt.show()

print(torch.mean(model_glif.neuron_layer.weight_iv))
with torch.no_grad():
    nn.init.constant_(model_glif.neuron_layer.weight_iv, 0.01)
    plot_responses(model_glif)
