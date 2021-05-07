# author: @chloewin
# 03/07/21
import pickle
import datetime
import utils as ut
from networks import RBNN
from neurons.glif_new import GLIFR, RNNC, Placeholder

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as tud
import torch.nn as nn
import math
# import torch.utils.data.DataLoader


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
	base_name = "figures_wkof_050221/init_attempt_5"
	# Generate freqs
	num_freqs = 6
	freq_min = 0.001
	freq_max = 0.6

	freqs = 10 ** np.linspace(np.log10(freq_min), np.log10(freq_max), num=num_freqs)

	# Generate data
	sim_time = 4.0
	dt = 0.05
	amp = 1.0
	noise_mean = 0
	noise_std = 0

	batch_size = 3

	inputs, targets = ut.create_sines(sim_time, dt, amp, noise_mean, noise_std, freqs)
	traindataset = ut.create_dataset(inputs, targets)

	# Generate model
	delay = int(1 / dt)
	model = RBNN(in_size = 1, hid_size = 500, out_size = 1, dt = dt, delay = delay)

	# Train model
	num_epochs = 250
	lr = 0.005
	reg_lambda = 1500
	training_info = ut.train_rbnn(model, traindataset, batch_size, num_epochs, lr, reg_lambda)

	colors = ["sienna", "peru", "peachpuff", "salmon", "red", "darkorange", "purple", "fuchsia", "plum", "darkorchid", "slateblue", "mediumblue", "cornflowerblue", "skyblue", "aqua", "aquamarine", "springgreen", "green"]

	# Plot outputs

	final_outputs = training_info["final_outputs"]

	for i in range(num_freqs):
		plt.plot(np.arange(len(final_outputs[i][0])) * dt, final_outputs[i][0,:,0,0].detach().numpy(), c = colors[i], label=f"freq {freqs[i]}")
		plt.plot(np.arange(len(final_outputs[i][0])) * dt, targets[:, i], '--', c = colors[i])
	plt.legend()
	plt.savefig("figures/" + base_name + "_final_outputs")
	plt.close()

	final_outputs_driven = training_info["final_outputs_driven"]
	for i in range(num_freqs):
		plt.plot(np.arange(len(final_outputs_driven[i][0])) * dt, final_outputs_driven[i][0,:,0,0].detach().numpy(), c = colors[i], label=f"freq {freqs[i]}")
		plt.plot(np.arange(len(final_outputs_driven[i][0])) * dt, targets[:, i], '--', c = colors[i])
	plt.legend()
	plt.xlabel("time (ms)")
	plt.ylabel("firing rate (1/ms)")
	plt.savefig("figures/" + base_name + "_final_outputs_driven")
	plt.close()

	init_outputs = training_info["init_outputs"]
	for i in range(num_freqs):
		plt.plot(np.arange(len(init_outputs[i][0])) * dt, init_outputs[i][0,:,0,0].detach().numpy(), c = colors[i], label=f"freq {freqs[i]}")
		plt.plot(np.arange(len(init_outputs[i][0])) * dt, targets[:, i], '--', c = colors[i])
	plt.legend()
	plt.xlabel("time (ms)")
	plt.ylabel("firing rate (1/ms)")
	plt.savefig("figures/" + base_name + "_init_outputs")
	plt.close()

	init_outputs_driven = training_info["init_outputs_driven"]
	for i in range(num_freqs):
		plt.plot(np.arange(len(init_outputs_driven[i][0])) * dt, init_outputs_driven[i][0,:,0,0].detach().numpy(), c = colors[i], label=f"freq {freqs[i]}")
		plt.plot(np.arange(len(init_outputs_driven[i][0])) * dt, targets[:, i], '--', c = colors[i])
	plt.legend()
	plt.xlabel("time (ms)")
	plt.ylabel("firing rate (1/ms)")
	plt.savefig("figures/" + base_name + "_init_outputs_driven")
	plt.close()

	# Plot losses
	plt.plot(training_info["losses"])
	plt.xlabel("epoch #")
	plt.ylabel("loss")
	plt.savefig("figures/" + base_name + "_losses")

	
	with open('filename.pickle', 'wb') as handle:
		pickle.dump(training_info, handle)

if __name__ == '__main__':
	main()