# author: @chloewin
# 03/07/21
import utils_train_new as ut
from neurons.network_new import SNNNetwork, RSNNNetwork, BNN3L, RNN
from neurons.glif_new import GLIFR, RNNC, Placeholder

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as tud
import torch.nn as nn


"""
This file trains a network of RNN neurons with after-spike currents on a pattern generation task.
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
	base_name = "sussillo_debug_rnn_tanh_debug_dd"
	fig_name = "figures/figures_wkof_042521/" + base_name#sussillo_newarchitecture_batch_agn_agn_agn_agn_agn"#singlepattern_noise_unifasckandtrainothers_agn-lnger-agn-noclip-noaareg"
	model_name = "saved_models/models_wkof_042521/" + base_name
	date = ""

	## Neuron specifications
	dt = 0.05
	sim_time = 3
	spike_r = 20
	
	steps = int(sim_time / dt)
	delay = 1#int(1 / dt)
	input_size = 1

	## Training specifications
	num_epochs = 5
	lr = 0.5
	batch_size = 2
	rnn=False

	km_reg_lambda = 1000
	train_params = ["weight", "bias"]#["thresh", "asc_r", "asc_amp", "ln_k_m"]

	str = "sussillo"

	single = (str == 'single')
	sussillo = (str == 'sussillo')
	bellec = (str == 'bellec')
	print(f"single: {single}")
	print(f"sussillo: {sussillo}")
	print(f"bellec: {bellec}")

	if single:
		batch_size = 1

	# Sussillo
	if sussillo:
		input_amplitudes = [-0.5,0.5]#, 125, 150, 175, 200]#, 225, 250, 275, 300]
		# input_amplitudes = [ia * 1000 for ia in input_amplitudes]
		# frequencies = [0.25, 0.5, 0.75, 1]
		frequencies = [0.01, 0.41, .81, 1.21]
		frequencies = [0.002, 0.008, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]#, 0.6, 0.8, 1.0, 1.2]
		frequencies = [0.25, 0.75]
		# frequencies = [0.01, 0.03, 0.05, 0.07]
		amplitudes = [2, 2]#, 2, 2, 2, 2]#, 1, 1, 1, 1]

		input_spikes_init = []
		target_spikes_init = []
		for i in range(len(input_amplitudes)):
			input_spikes_init.append(input_amplitudes[i] * torch.ones((steps, 1, 1, input_size)))
			# input_spikes_init.append(input_amplitudes[i] * ut.generate_sinusoid(spike_r, frequencies[i], sim_time, dt, amplitude = amplitudes[i]).reshape(steps, 1, 1, input_size))
			target_spikes_init.append(ut.generate_sinusoid(spike_r, frequencies[i], sim_time, dt, amplitude = amplitudes[i]).reshape(steps, 1, 1, 1))
	
		sussillo_dataset = tud.TensorDataset(torch.stack(input_spikes_init, dim=0), torch.stack(target_spikes_init, dim=0))
		dataloader = tud.DataLoader(sussillo_dataset, batch_size=batch_size, shuffle=True)
	# Bellec
	if bellec:
		input_amplitudes = [500, 500, 500, 500, 500, 500, 500, 500]
		# frequencies = [0.01, 0.03, 0.05, 0.07]
		frequencies = [0.002, 0.008, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
		# amplitudes = [1, 1, 1, 1]
		amplitudes = [2, 2, 2, 2, 2, 2, 2, 2]

		n_parts = len(frequencies)
		input_spikes_init = []
		target_spikes_init = []

		for i in range(len(input_amplitudes)):
			isi = (torch.zeros((steps, 1, 1, input_size)))
			isi[:,:,:,i * int(input_size / n_parts):(i+1) * int(input_size / n_parts)] = input_amplitudes[i]
			input_spikes_init.append(isi)
			target_spikes_init.append(ut.generate_sinusoid(spike_r, frequencies[i], sim_time, dt, amplitude = amplitudes[i]).reshape(steps, 1, 1, 1))
		bellec_dataset = tud.TensorDataset(torch.stack(input_spikes_init, dim=0), torch.stack(target_spikes_init, dim=0))
		dataloader = tud.DataLoader(bellec_dataset, batch_size=batch_size)

	sizes = {"#input": input_size,
			 "hidden_1": 150,
			 # "hidden_2": 100,
			 "##average": 1
	}
	layers = [
		"#input",
		"hidden_1",
		# "hidden_2",
		"##average"
	]
	conns = {
		"#input_for": ("#input", "hidden_1"),
		# "hidden1_for": ("hidden_1", "hidden_2"),
		# # "hidden2_rec": ("hidden_2", "hidden_1"),
		"hidden1_for": ("hidden_1", "##average")
	}

	input_size = sizes[layers[0]]

	## Construct Input
	if single:
		input_spikes = torch.ones((steps, 1, 1, input_size))
		plt.plot(np.arange(len(input_spikes)) * dt, input_spikes[:,0,0,0], color = "deepskyblue")
		plt.xlabel('time (ms)')
		plt.ylabel('spiking output (1/ms)')
		plt.savefig(fig_name + "_inputs_" + date)
		plt.close()

	## Construct Target
	if single:
		freq = 1
		target_spikes = ut.generate_sinusoid(spike_r, freq, sim_time, dt).reshape((steps, 1, 1, 1)) #20 * torch.ones((steps, 1, 1, architecture[0]))#
		plt.plot(np.arange(len(target_spikes)) * dt, target_spikes.detach()[:,0,0,0], color = "indianred", label = "target")

	## Create Training Network
	train_network = RNN(layers, sizes, conns, delay = delay, dt=dt, batch_size=batch_size)
	# for i in [-0.5,-0.25,0,0.25,0.5]:
	# 	train_network.reset_state(batch_size=1)
	# 	input_spikes =  i * torch.ones(steps, 1, 1, input_size) 
	# 	# input_spikes = input_spikes + torch.randn_like(input_spikes)
	# 	# input_spikes = i * ut.generate_sinusoid(spike_r, 0.01, sim_time, dt, amplitude = 10).reshape(steps, 1, 1, input_size)
	# 	train_spikes = train_network.forward(input_spikes)
	# 	print( torch.stack(train_spikes, dim=0).shape)
	# 	# train_spikes = torch.stack(train_network._layers["hidden"].voltages, dim=0)
	# 	# train_spikes = torch.mean(train_spikes, dim=-1).detach()
	# 	# plt.plot(np.arange(len(ts) - delay) * dt, train_spikes[delay:].reshape(len(ts)-delay), label = f"{i}")
	# 	plt.plot(np.arange(len(train_spikes) - delay) * dt, torch.stack(train_spikes, dim=0).detach()[delay:, 0,0,:], label = f"{i}")
	# plt.legend()
	# plt.show()

	if single:
		init_spikes = train_network.forward(input_spikes)
		init_spikes_all = {}
		for name in layers:
			# print(len(train_network._layers[name].spikes_all))
			init_spikes_all[name] = torch.stack(train_network._layers[name].spikes_all, dim=0)

		colors = ["palegreen", "yellowgreen", "mediumseagreen", "darkgreen", "yellow"]
		c = -1
		for name in layers:
			c += 1
			for i in range(sizes[name]):
				# print(sizes[name])
				# print(init_spikes_all[name].detach().shape)
				plt.plot(np.arange(len(init_spikes) - delay) * dt, init_spikes_all[name].detach()[delay:,0,0,i], color = colors[c], label = name if i == 0 else "")
				if not ('average' in layers or '#average' in layers):
					# c += 1
					plt.plot(np.arange(len(init_spikes) - delay) * dt, torch.mean(init_spikes_all[layers[-1]].detach()[delay:, 0,0,:],-1), color = colors[c], label = "average")
		plt.legend()
		plt.xlabel('time (ms)')
		plt.ylabel('spiking output (1/ms)')
		plt.savefig(fig_name + "_init_" + date)
		plt.close()

	## Training
	# Generate dictionaries
	train_dict = {}
	state_dicts = {}

	tracker = {}
	tracker['losses'] = []
	for i in range(len(layers)):
		name = layers[i]
		tracker[name + "_activation"] = []

		if name[0] != '#' or name[1] == '#':
			for p in train_params:
				tracker[name + "_" + p] = []
				tracker[name + "_" + p + "_grad"] = []

	optimizer = optim.Adam(train_network.parameters(), lr=lr)
	
	## Training loop
	train_spikes_all = {}
	train_spikes = 0

	tsa = {}
	ts = 0

	for ep in range(num_epochs):
		# rnn = False#(1==1) #ep < 200:
		optimizer.zero_grad()
		train_network.reset_state()
		loss = 0.0

		if sussillo or bellec:
			# with torch.no_grad():
			# 	pass
				# i_rand = ep % len(input_amplitudes)#random.randint(0, n_parts - 1)
				# if ep < 100:
				# 	i_rand = 0
				# input_spikes = input_spikes_init[i_rand]
				# target_spikes = target_spikes_init[i_rand]
			batch_count = 0
			for batch_ndx, sample in enumerate(dataloader):
				optimizer.zero_grad()
				loss = 0
				batch_count += 1
				train_network.reset_state(batch_size=len(sample))
				with torch.no_grad():
					input_spikes, target_spikes = sample
					a, b, c, d, e = input_spikes.shape
					input_spikes = input_spikes.reshape(b, a, d, e)

					input_spikes_randn = input_spikes + torch.randn_like(input_spikes)

					a, b, c, d, e = target_spikes.shape
					target_spikes = target_spikes.reshape(b, a, d, e)

					target_spikes_randn = target_spikes + torch.randn_like(target_spikes) * 0.001

				ts = train_network.forward(input_spikes_randn)
				# print(torch.stack(ts).shape)
				# ts = torch.stack(ts, dim=0)
				for name in layers:
					tsa[name] = torch.stack(train_network._layers[name].spikes_all, dim=0)

				for j in range(steps):
					loss = loss + ut.avg_norm(ts[j + delay], target_spikes_randn[j,:,:])
				# loss = loss / batch_count
				if batch_ndx < len(dataloader) - 1:
					loss = loss + ut.km_reg(train_network, km_reg_lambda)
					loss.backward()
					optimizer.step()
		else:
			ts = train_network.forward(input_spikes)
			for name in layers:
				tsa[name] = torch.stack(train_network._layers[name].spikes_all, dim=0)

			target_spikes_randn = target_spikes + torch.randn_like(target_spikes) * 0.001

			for j in range(steps):
				loss = loss + ut.avg_norm(ts[j + delay], target_spikes_randn[j,:,:])
		# loss = loss + ut.fft_norm(torch.stack(ts[delay:], dim=0), target_spikes)
		# loss = loss + ut.km_reg(train_network, km_reg_lambda) ##TODO: CHANGE back to 50
		# loss = loss + ut.asc_amp_reg(train_network, 5)
		# loss = loss + ut.asc_k_reg(train_network)

		if torch.isnan(loss):
			num_epochs = ep
			print("loss is nan")
			break
		# if single:
		train_spikes = ts
		train_spikes_all = tsa

		# Printing and tracking
		print(f"...loss at epoch {ep}: {loss.data}")
		tracker['losses'].append(loss.data)
		for i in range(len(layers)):#name, layer in target_network._layers.items():
			name = layers[i]

			if name[0] != '#' or name[1] == '#':
				layer = train_network._layers[name]
				if "thresh" in train_params:
					tracker[name + "_thresh"].append([layer.thresh[0,0,j] + 0.0 for j in range(sizes[name])])
				if "ln_k_m" in train_params:
					tracker[name + "_ln_k_m"].append([torch.exp(layer.ln_k_m[0,0,j])  + 0.0 for j in range(sizes[name])])
				if "asc_amp" in train_params:
					tracker[name + "_asc_amp"].append([layer.asc_amp[j,0,0,m]  + 0.0 for j in range(len(layer.asc_amp)) for m in range(sizes[name])])
				if "asc_r" in train_params:
					tracker[name + "_asc_r"].append([layer.asc_r[j,0,0,m] + 0.0 for j in range(len(layer.asc_r)) for m in range(sizes[name])])
				if "asc_k" in train_params:
					tracker[name + "_asc_k"].append([torch.exp(layer.asc_k[j,0,0,m]) + 0.0 for j in range(len(layer.asc_k)) for m in range(sizes[name])])
				if "weight" in train_params:
					s1, s2 = layer.linear.weight.shape
					tracker[name + "_weight"].append([layer.linear.weight[j, i] + 0.0 for j in range(s1) for i in range(s2)])
				if "bias" in train_params:
					tracker[name + "_bias"].append([layer.linear.bias[i] + 0.0 for i in range(s1)])
		
		# Update
		loss.backward()

		for i in range(len(layers)):#name, layer in target_network._layers.items():
			name = layers[i]
			tracker[name + "_activation"].append(torch.mean(train_spikes_all[name], dim=[0,1,2,3]).item())
			# name = layers[i]
			layer = train_network._layers[name]
			if name[0] != '#':
				if "thresh" in train_params and not rnn:
					tracker[name + "_thresh_grad"].append([layer.thresh.grad[0,0,j] + 0.0 for j in range(sizes[name])])
				if "ln_k_m" in train_params and not rnn:
					tracker[name + "_ln_k_m_grad"].append([(layer.ln_k_m.grad[0,0,j])  + 0.0 for j in range(sizes[name])])
				if "asc_amp" in train_params:
					tracker[name + "_asc_amp_grad"].append([layer.asc_amp.grad[j,0,0,m] for j in range(len(layer.asc_amp)) for m in range(sizes[name])])
				if "asc_r" in train_params and not rnn:
					tracker[name + "_asc_r_grad"].append([layer.asc_r.grad[j,0,0,m] for j in range(len(layer.asc_r)) for m in range(sizes[name])])
				if "asc_k" in train_params and not rnn:
					tracker[name + "_asc_k_grad"].append([layer.asc_k.grad[j,0,0,m] for j in range(len(layer.asc_k)) for m in range(sizes[name])])
				if "weight" in train_params:
					s1, s2 = layer.linear.weight.shape
					tracker[name + "_weight_grad"].append([layer.linear.weight.grad[j, i] + 0.0 for j in range(s1) for i in range(s2)])
				if "bias" in train_params:
					tracker[name + "_bias_grad"].append([layer.linear.bias.grad[i] + 0.0 for i in range(s1)])
				# if rnn:#ep < 200:
				# 	with torch.no_grad():
				# 		# layer.asc_k.grad *= 0
				# 		# layer.asc_r.grad *= 0
				# 		# layer.asc_amp.grad *= 0
				# 		# layer.ln_k_m.grad *= 0
				# 		layer.thresh.grad *= 0
				# else:
				# 	layer.asc_k.grad *= 0
				# 	# layer.ln_k_m.grad *= 0
				# # layer.asc_r.grad *= 0
				# # layer.asc_amp.grad *= 0
				# for name in conns.keys():
				# 	pre, post = conns[name]
				# 	layer = train_network._conns[name]
				# 	tracker[name + "_weight_grad"].append([layer.weight.grad[j,0] + 0.0 for j in range(sizes[post])])

		optimizer.step()
	# Finsih plotting spikes_all
	if single:
		colors = ["lightpink", "thistle", "magenta", "saddlebrown"]
		c = -1
		for name in layers:
			# if name[0] != '#':
			c += 1
			for i in range(sizes[name]):
				plt.plot(np.arange(len(train_spikes) - delay) * dt, train_spikes_all[name].detach()[delay:,0,0,i], color = colors[c], label = name if i == 0 else "")
				c += 1
				if ('average' not in layers and '#average' not in layers):
					plt.plot(np.arange(len(train_spikes) - delay) * dt, torch.mean(train_spikes_all[layers[-1]].detach()[delay:, 0,0,:],-1), color = colors[c], label = "average")

					plt.plot(np.arange(len(target_spikes)) * dt, target_spikes.detach()[:,0,0,0], color = "yellow", label = "target")

		plt.legend()
		plt.xlabel('time (ms)')
		plt.ylabel('spiking output (1/ms)')
		plt.savefig(fig_name + "_spikes_all_" + date)
		plt.xlim(0,10)
		plt.savefig(fig_name + "_spikes_all_zoom_" + date)
		plt.xlim(30,40)
		plt.savefig(fig_name + "_spikes_all_zoom_end_" + date)
		plt.close()

	if sussillo or bellec:
		colors = ["orangered", "darkkhaki", "yellowgreen", "cyan", "deepskyblue", "crimson", "magenta", "lawngreen", "darkviolet", "sienna"]
		colors = ["sienna", "peru", "peachpuff", "salmon", "red", "darkorange", "purple", "fuchsia", "plum", "darkorchid", "slateblue", "mediumblue", "cornflowerblue", "skyblue", "aqua", "aquamarine", "springgreen", "green"]
		c = -1
		for j in range(len(input_amplitudes)):
			c += 1
			train_network.reset_state(batch_size=1)
			with torch.no_grad():
				input_spikes = input_spikes_init[j]#torch.zeros((steps, 1, 1, input_size))
				target_spikes = target_spikes_init[j]

				train_spikes = train_network.forward(input_spikes)
				# for name in layers:
				# 	train_spikes_all[name] = torch.stack(train_network._layers[name].spikes_all, dim=0)
				train_spikes = torch.stack(train_spikes, dim=0)
				plt.plot(np.arange(len(train_spikes) - delay) * dt, torch.mean(train_spikes.detach()[delay:, 0,0,:],-1), color = colors[c], label = f"predicted {frequencies[j]} Hz")
				plt.plot(np.arange(len(target_spikes)) * dt, target_spikes.detach()[:,0,0,0], '--', color = colors[c])
		plt.xlabel('time (ms)')
		plt.ylabel('spiking output (1/ms)')
		plt.legend()
		plt.savefig(fig_name + "_spikes_all_all_" + date)
		plt.xlim([0,10])
		plt.savefig(fig_name + "_spikes_all_all_zoom_left" + date)
		plt.xlim([30,40])
		plt.savefig(fig_name + "_spikes_all_all_zoom_right" + date)
		plt.close()

	# Plot average activation over epocsh
	c = -1
	for name in layers:
		c += 1
		for i in range(sizes[name]):
			diff = np.array(tracker[name + "_activation"]) 
			plt.plot(range(num_epochs), diff, color = colors[c], label = name if i ==0 else "")
	plt.legend()
	plt.xlabel('# epoch')
	plt.ylabel('average output')
	plt.savefig(fig_name + "_activations_" + date)
	plt.close()

	# Plot parameters
	colors = ["lightcoral", "rosybrown", "aquamarine", "brown", "tomato", "darkorange", "goldenrod", "olivedrab", "darkcyan", "violet", "blue", "skyblue", "fuchsia", "mediumspringgreen", "gold", "lightpink", "yellow", "darkmagenta"]

	n = -1
	for i in range(len(layers)):
		name = layers[i]
		if name[0] != '#' or name[1] == '#':
			list = []
			for p in train_params:
				list.append(name + "_" + p)
			for k in list:
				diff = np.array(tracker[k]) 
				_, l = diff.shape
				n += 1
				for j in range(l):
					plt.plot(range(num_epochs), diff[:,j], color = colors[n], label = k if j == 0 else "")
	plt.axhline(y=0, color='k', linestyle='-')
	plt.legend()
	plt.xlabel("# epoch")
	plt.ylabel("learned")
	# plt.show()
	plt.savefig(fig_name + "_params_" + date)
	plt.close()

	# Plot parameter gradients
	n = -1
	for i in range(len(layers)):
		name = layers[i]

		if name[0] != '#':
			list = []
			for p in train_params:
				list.append(name + "_" + p + "_grad")
			for k in list:
				diff = np.array(tracker[k]) 
				_, l = diff.shape
				n += 1
				for j in range(l):
					plt.plot(range(num_epochs), diff[:,j], color = colors[n], label = k if j == 0 else "")
	plt.axhline(y=0, color='k', linestyle='-')
	plt.legend()
	plt.xlabel("# epoch")
	plt.ylabel("gradient")
	plt.savefig(fig_name + "_param-grads_" + date)
	plt.close()

	# Plot losses
	plt.plot(range(num_epochs), tracker['losses'])
	plt.axhline(y=0, color='k', linestyle='-')
	plt.xlabel("# epoch")
	plt.ylabel("loss")
	plt.savefig(fig_name + "_losses_" + date)
	plt.close()

	torch.save(train_network.state_dict(), model_name + '.pt')

if __name__ == '__main__':
	main()