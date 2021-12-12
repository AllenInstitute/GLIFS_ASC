"""
This file is used for miscellaneous plotting and analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud
import math
from models.networks import RNNFC, BNNFC, LSTMFC
import utils_task as utta 

fontsize = 18

# NOTE: must change below lines to point to appropriate files
main_name = "smnist-4-final"
base_name_results = "test/" + main_name
base_name_model = "test/" + main_name

init = True
input_size = 28 # 1 for pattern
hid_size = 256 # 128 for pattern
output_size = 10 # 1 for pattern

ficurve_simtime = 5

def plot_example_steps():
    # Simulates single neuron responding to constant input of different magnitudes
    filename = "sample-outputs-steps-sigmav1-3"
    filename_dir = "results_wkof_080821/" + filename

    sigma_v = 1e-3 # 1
    scale_factors = [0.1, 1, 10]

    sim_time = 40
    dt = 0.05
    nsteps = int(sim_time / dt)

    input_size = 1
    output_size = 1
    hid_size = 1

    inputs = torch.ones(1, nsteps, input_size)
    # Uncomment next two lines to explore step-like input rather than constant input
    # inputs = torch.zeros(1, nsteps, input_size)
    # inputs[:, 100:700, :] = 1

    model_glif = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, output_weight=False)
    with torch.no_grad():
        model_glif.neuron_layer.sigma_v = sigma_v
    asc_rs = (-(1-1e-10), 1-1e-10)
    asc_amps = (50, -50)
    asc_ks = (.2, .2)
    
    model_glif.eval()
    for i in range(len(scale_factors)):
        scale_factor = scale_factors[i]
        inputs_here = inputs * scale_factor

        model_glif.reset_state(1)
        with torch.no_grad():
            model_glif.neuron_layer.weight_iv.data = torch.ones((input_size, hid_size))
            model_glif.neuron_layer.weight_lat.data = torch.zeros((hid_size, hid_size))
            model_glif.neuron_layer.thresh.data *= 0

            asc_r1, asc_r2 = asc_rs
            asc_amp1, asc_amp2 = asc_amps
            asc_k1, asc_k2 = asc_ks

            model_glif.neuron_layer.trans_asc_r[0,0,0] = math.log((1 - asc_r1) / (1 + asc_r1))
            model_glif.neuron_layer.trans_asc_r[1,0,0] = math.log((1 - asc_r2) / (1 + asc_r2))

            model_glif.neuron_layer.asc_amp[0,0,0] = asc_amp1
            model_glif.neuron_layer.asc_amp[1,0,0] = asc_amp2

            model_glif.neuron_layer.trans_asc_k[0,0,0] = math.log(asc_k1 * dt / (1 - (asc_k1 * dt))) 
            model_glif.neuron_layer.trans_asc_k[1,0,0] = math.log(asc_k2 * dt / (1 - (asc_k2 * dt))) 

        outputs, voltages, ascs, syns = model_glif.forward(inputs_here, track=True)
        outputs = outputs.detach().numpy()
        voltages = voltages.detach().numpy()
        ascs = ascs.detach().numpy()
        syns = syns.detach().numpy()

        np.savetxt("results/" + filename_dir + "-" + str(i) + ".csv", outputs[0,:,0], delimiter=",")
        np.savetxt("results/" + filename_dir + "-" + str(i) + "syn.csv", syns[0,:,0], delimiter=",")
        np.savetxt("results/" + filename_dir + "-" + str(i) + "voltage.csv", voltages[0,:,0], delimiter=",")
        np.savetxt("results/" + filename_dir + "-" + str(i) + "asc.csv", ascs[:,0,:,0], delimiter=",")
        np.savetxt("results/" + filename_dir + "-" + str(i) + "in.csv", inputs_here[0,:,0], delimiter=",")

        plt.plot(outputs[0,:,0], label=str(scale_factor))
    plt.legend()
    plt.show()

def plot_example_step():
    # Simulates single neuron responding to step-like current
    filename = "sample-outputs-step"
    filename_dir = "results_wkof_080821/" + filename
    name = filename

    sim_time = 40
    dt = 0.05
    nsteps = int(sim_time / dt)

    input_size = 1
    output_size = 1
    hid_size = 1

    inputs = torch.zeros(1, nsteps, input_size)
    inputs[:, 100:400, :] = 1

    model_glif = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, output_weight=False)
    asc_r = (-(1 - 1e-10), -(1 - 1e-10))
    asc_amp = (-5000, -5000)
    asc_k = (0.5, 0.5)
    
    model_glif.reset_state(1)
    with torch.no_grad():
        model_glif.neuron_layer.weight_iv.data = torch.ones((input_size, hid_size))
        model_glif.neuron_layer.weight_lat.data = torch.zeros((hid_size, hid_size))
        model_glif.neuron_layer.thresh.data *= 0

        asc_r1, asc_r2 = asc_r
        asc_amp1, asc_amp2 = asc_amp
        asc_k1, asc_k2 = asc_k

        model_glif.neuron_layer.trans_asc_r[0,0,0] = math.log((1 - asc_r1) / (1 + asc_r1))
        model_glif.neuron_layer.trans_asc_r[1,0,0] = math.log((1 - asc_r2) / (1 + asc_r2))

        model_glif.neuron_layer.asc_amp[0,0,0] = asc_amp1
        model_glif.neuron_layer.asc_amp[1,0,0] = asc_amp2

        model_glif.neuron_layer.trans_asc_k[0,0,0] = math.log(asc_k1 * dt / (1 - (asc_k1 * dt))) 
        model_glif.neuron_layer.trans_asc_k[1,0,0] = math.log(asc_k2 * dt / (1 - (asc_k2 * dt))) 

    outputs, voltages, ascs, syns = model_glif.forward(inputs, track=True)
    outputs = outputs.detach().numpy()
    voltages = voltages.detach().numpy()
    ascs = ascs.detach().numpy()
    syns = syns.detach().numpy()

    np.savetxt("results/" + filename_dir + "-" + name + ".csv", outputs[0,:,0], delimiter=",")
    np.savetxt("results/" + filename_dir + "-" + name + "syn.csv", syns[0,:,0], delimiter=",")
    np.savetxt("results/" + filename_dir + "-" + name + "voltage.csv", voltages[0,:,0], delimiter=",")
    np.savetxt("results/" + filename_dir + "-" + name + "asc.csv", ascs[:,0,:,0], delimiter=",")
    np.savetxt("results/" + filename_dir + "-" + name + "in.csv", inputs[0,:,0], delimiter=",")

    plt.plot(outputs[0,:,0], label=name)
    plt.legend()
    plt.show()

def plot_example_step_rnn():
    # Simulate response of RNN to step-like input
    filename = "sample-outputs-step-rnn"
    filename_dir = "results_wkof_080821/" + filename
    name = filename

    sim_time = 40
    dt = 0.05
    nsteps = int(sim_time / dt)

    input_size = 1
    output_size = 1
    hid_size = 1

    inputs = torch.zeros(1, nsteps, input_size)
    inputs[:, 100:400, :] = 10

    model_rnn = RNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, output_weight=False)
    
    model_rnn.reset_state(1)
    with torch.no_grad():
        model_rnn.neuron_layer.weight_ih.data = torch.ones((input_size, hid_size))
        model_rnn.neuron_layer.weight_lat.data = torch.zeros((hid_size, hid_size))

    outputs, voltages = model_rnn.forward(inputs, track=True)
    outputs = outputs.detach().numpy()
    voltages = voltages.detach().numpy()

    np.savetxt("results/" + filename_dir + "-" + name + ".csv", outputs[0,:,0], delimiter=",")
    np.savetxt("results/" + filename_dir + "-" + name + "voltage.csv", voltages[0,:,0], delimiter=",")
    np.savetxt("results/" + filename_dir + "-" + name + "in.csv", inputs[0,:,0], delimiter=",")

    plt.plot(outputs[0,:,0], label=name)
    plt.legend()
    plt.show()

def plot_examples():
    # Simulate response of several neurons to constant input
    filename = "sample-outputs-sigmav1e-3"
    filename_dir = "results_wkof_080821/" + filename

    sim_time = 40
    dt = 0.05
    nsteps = int(sim_time / dt)

    input_size = 1
    output_size = 1
    hid_size = 1

    inputs = torch.ones(1, nsteps, input_size)

    model_glif = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, output_weight=False)
    asc_rs = [(0,0), (-(1 - 1e-10), -(1 - 1e-10)), (-(1 - 1e-10), (1 - 1e-10))]
    asc_amps = [(0, 0), (-5000, -5000), (5000, -5000)]
    asc_ks = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]
    names = ["zero", "neg", "opp"]
    
    for i in range(len(asc_rs)):
        model_glif.reset_state(1)
        with torch.no_grad():
            model_glif.neuron_layer.weight_iv.data = torch.ones((input_size, hid_size))
            model_glif.neuron_layer.weight_lat.data = torch.zeros((hid_size, hid_size))
            model_glif.neuron_layer.thresh.data *= 0
            name = names[i]

            asc_r1, asc_r2 = asc_rs[i]
            asc_amp1, asc_amp2 = asc_amps[i]
            asc_k1, asc_k2 = asc_ks[i]

            model_glif.neuron_layer.trans_asc_r[0,0,0] = math.log((1 - asc_r1) / (1 + asc_r1))
            model_glif.neuron_layer.trans_asc_r[1,0,0] = math.log((1 - asc_r2) / (1 + asc_r2))

            model_glif.neuron_layer.asc_amp[0,0,0] = asc_amp1
            model_glif.neuron_layer.asc_amp[1,0,0] = asc_amp2

            model_glif.neuron_layer.trans_asc_k[0,0,0] = math.log(asc_k1 * dt / (1 - (asc_k1 * dt))) 
            model_glif.neuron_layer.trans_asc_k[1,0,0] = math.log(asc_k2 * dt / (1 - (asc_k2 * dt))) 

        outputs, voltages, ascs, syns = model_glif.forward(inputs, track=True)
        outputs = outputs.detach().numpy()
        voltages = voltages.detach().numpy()
        ascs = ascs.detach().numpy()
        syns = syns.detach().numpy()

        np.savetxt("results/" + filename_dir + "-" + name + ".csv", outputs[0,:,0], delimiter=",")
        np.savetxt("results/" + filename_dir + "-" + name + "syn.csv", syns[0,:,0], delimiter=",")
        np.savetxt("results/" + filename_dir + "-" + name + "voltage.csv", voltages[0,:,0], delimiter=",")
        np.savetxt("results/" + filename_dir + "-" + name + "asc.csv", ascs[:,0,:,0], delimiter=",")
        np.savetxt("results/" + filename_dir + "-" + name + "in.csv", inputs[0,:,0], delimiter=",")

        plt.plot(outputs[0,:,0], label=name)
    plt.legend()
    plt.show()


def plot_ficurve(model):
    # Produces firing rates and input currents needed for a f-I curve

    sim_time = ficurve_simtime
    dt = 0.05
    nsteps = int(sim_time / dt)

    i_syns = np.arange(-10000, 10000, step=100)

    input = torch.zeros(1, nsteps, input_size)

    f_rates = np.zeros((len(i_syns), hid_size))
    for i in range(len(i_syns)):
        firing = torch.zeros((input.shape[0], hid_size))
        voltage = torch.zeros((input.shape[0], hid_size))
        syncurrent = torch.zeros((input.shape[0], hid_size))
        ascurrents = torch.zeros((2, input.shape[0], hid_size))
        outputs_temp = torch.zeros(1, nsteps, hid_size)

        firing_delayed = torch.zeros((input.shape[0], nsteps, hid_size))

        model.neuron_layer.I0 = i_syns[i]
        for step in range(nsteps):
            x = input[:, step, :]
            firing, voltage, ascurrents, syncurrent = model.neuron_layer(x, firing, voltage, ascurrents, syncurrent, firing_delayed[:, step, :])
            outputs_temp[0, step, :] = firing
        f_rates[i, :] = torch.mean(outputs_temp, 1).detach().numpy().reshape((1, -1))

    print(f"f_rates.shape = {f_rates.shape}")

    slopes = []#np.zeros(hid_size)
    for i in range(hid_size):
        i_syns_these = i_syns
        f_rates_these = f_rates[:,i]
        indices = np.logical_not(np.logical_or(np.isnan(i_syns_these), np.isnan(f_rates_these)))     
        indices = np.array(indices)
        i_syns_these = i_syns_these[indices]
        
        f_rates_these = f_rates_these[indices] #* sim_time / dt
        i_syns_these = i_syns_these[f_rates_these > 0.01]
        f_rates_these = f_rates_these[f_rates_these > 0.01] * sim_time / dt


        A = np.vstack([i_syns_these, np.ones_like(i_syns_these)]).T
        m, c = np.linalg.lstsq(A, f_rates_these)[0]
        if len(f_rates_these) > 0:
            slopes.append(m)

        if m < 0:
            print(f"found negative slope in neuron {i}")
            print(f_rates_these)
            print(m)
        plt.plot(i_syns_these, f_rates_these)
    np.savetxt("results/" + base_name_results + "-" + ("init-" if init else "") + "slopes.csv", np.array(slopes), delimiter=",")
    np.savetxt("results/" + base_name_results + "-" + ("init-" if init else "") + "isyns.csv", np.array(i_syns), delimiter=",")
    np.savetxt("results/" + base_name_results + "-" + ("init-" if init else "") + "frates.csv", np.array(f_rates), delimiter=",")

def plot_pattern_responses():
    base_name = "saved_models/models_wkof_080821"
    filenames = ["pattern-1-131units-5itr.pt", "pattern-2-128units-0itr.pt", "pattern-3-131units-0itr.pt", "pattern-4-128units-5itr.pt", "pattern-5-131units-5itr.pt", "pattern-6-130units-5itr.pt", "pattern-7-131units-5itr.pt", "pattern-8-130units-5itr.pt", "pattern-9-131units-5itr.pt", "pattern-10-64units-5itr.pt"]
    output_base_name = "results/results_wkof_080821"
    output_filenames = ["pattern-1-5itr", "pattern-2-0itr", "pattern-3-0itr", "pattern-4-5itr", "pattern-5-5itr", "pattern-6-5itr", "pattern-7-5itr", "pattern-8-5itr", "pattern-9-5itr", "pattern-10-5itr"]
    hid_sizes = [131, 128, 131, 128, 131, 130, 131, 130, 131, 64]

    # Task parameters
    sim_time = 5
    dt = 0.05
    nsteps = int(sim_time / dt)
    num_freqs = 6
    freq_min = 0.08
    freq_max = 0.6
    amp = 1
    noise_mean = 0
    noise_std = 0

    freqs = 10 ** np.linspace(np.log10(freq_min), np.log10(freq_max), num=num_freqs)

    inputs, targets = utta.create_sines_amp(sim_time, dt, amp, noise_mean, noise_std, freqs)
    traindataset = utta.create_dataset(inputs, targets)
    init_dataloader = tud.DataLoader(traindataset, batch_size=1, shuffle=False)
    
    hetinit_nums = [0, 1, 4, 5]
    learnparams_nums = [1, 3, 5, 7]
    for i in range(10):
        print(f"on {i}")
        model_glif = BNNFC(in_size = 1, hid_size = hid_sizes[i], out_size = 1, dt=dt, hetinit=(i in hetinit_nums), ascs=(i > 3), learnparams=(i in learnparams_nums))
        if i == 8:
            model_glif = RNNFC(in_size = 1, hid_size = hid_sizes[i], out_size = 1, dt=dt)
        elif i == 9:
            model_glif = LSTMFC(in_size = 1, hid_size = hid_sizes[i], out_size = 1, dt=dt)
        model_glif.load_state_dict(torch.load(base_name + "/" + filenames[i]))
        model_glif.eval()

        final_outputs = np.zeros((nsteps, len(init_dataloader)))
        final_targets = np.zeros((nsteps, len(init_dataloader)))
        for idx, sample in enumerate(init_dataloader):
            inputs, targets = sample
            _, nsteps, _ = inputs.shape
            model_glif.reset_state()
            outputs = model_glif.forward(inputs)
            outputs = outputs[0, -nsteps:, 0]
            targets = targets[0, -nsteps:, 0]

            final_outputs[:, idx] = outputs.detach().numpy()
            final_targets[:, idx] = targets.detach().numpy()
        np.savetxt(output_base_name + "/" + output_filenames[i] + "-targets.csv", final_targets, delimiter=',')
        np.savetxt(output_base_name + "/" + output_filenames[i] + "-learnedoutputs.csv", final_outputs, delimiter=',')

## Gather results for f-I curves
# model_glif = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size)

# if init:
#     model_glif.load_state_dict(torch.load("saved_models/" + base_name_model + "-" + str(hid_size) + "units-" + str(0) + "itr-init.pt"))
# else:
#     model_glif.load_state_dict(torch.load("saved_models/" + base_name_model + "-" + str(hid_size) + "units-" + str(0) + "itr.pt"))


# nn.init.constant_(model_glif.neuron_layer.weight_iv, 1)

# sim_time = ficurve_simtime
# dt = 0.05
# nsteps = int(sim_time / dt)

# input = torch.ones((1, nsteps, input_size))
# firing = torch.zeros((input.shape[0], hid_size))
# voltage = torch.zeros((input.shape[0], hid_size))
# syncurrent = torch.zeros((input.shape[0], hid_size))
# ascurrents = torch.zeros((2, input.shape[0], hid_size))
# outputs_temp = torch.zeros(1, nsteps, hid_size)

# firing_delayed = torch.zeros((input.shape[0], nsteps, hid_size))

# plot_ficurve(model_glif)

# Simply call any of the other functions
plot_pattern_responses()
