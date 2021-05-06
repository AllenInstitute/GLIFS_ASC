# author: @chloewin
# 03/07/21
import utils_train_new as ut
from neurons.network_new import SNNNetwork, RSNNNetwork, BNN3L, RNN3L
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
	base_name = "sussillo_debug_aaa_b_agn"
	fig_name = "figures/figures_wkof_042521/" + base_name#sussillo_newarchitecture_batch_agn_agn_agn_agn_agn"#singlepattern_noise_unifasckandtrainothers_agn-lnger-agn-noclip-noaareg"
	model_name = "saved_models/models_wkof_042521/" + base_name
	date = ""

	## Neuron specifications
	spike_r = 20
	sim_time = 4
	dt = 1 / spike_r
	steps = int(sim_time / dt)
	delay = int(1 / dt)
	input_size = 1

	## Training specifications
	num_epochs = 1500
	lr = 0.01
	batch_size = 1

	km_reg_lambda = 1500
	rnn=False
	train_params = ["thresh", "asc_r", "asc_amp", "ln_k_m"]

	sizes = {"#input": input_size,
		 # "l1": 100,
		 # "l2": 100,
		 # "input": input_size,
		 # "#average": 1
		 # "input": 5,
		 # "hidden1": 300,
		 # "hidden2": 300,
		 "hidden": 450,
		 # "hidden1": 100,#,
		 # "hidden2": 100,#,
		 "#average": 1
	}
	layers = [
		"#input",
		# "l1",
		# "l2",
		# "input", 
		# "#average"
		"hidden",
		# "hidden1", 
		# "hidden2",
		"#average"
	]
	conns = {
		# "input_for": ("input", "")
		# "input_rec": ("input", "input")
		# "input_in": ("#input", "input"),
		# "input_for": ("#input", "l1"),
		# "l1_for": ("l1", "l2"),
		# "l2_for": ("l2", "hidden"),
		# "#input_for": ("#input", "hidden1"),
		# "hidden1_for": ("hidden1", "hidden2"),
		# "hidden_rec": ("hidden2", "hidden1"),
		# "hidden2_for": ("hidden2", "#average"),
		"#input_for": ("#input", "hidden"),
		"hidden_rec": ("hidden", "hidden"),
		"hidden_for": ("hidden", "#average"),
		# "output_bac": ("#average", "hidden")
		# "hidden1_for": ("hidden1", "hidden2"),
		# "hidden2_rec": ("hidden2", "hidden1"),
		# "hidden2_out": ("hidden2", "#average"),
		# "output_rec": ("#average", "hidden"),
	}

	input_size = sizes[layers[0]]

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
		input_amplitudes = [250, 500]#, 125, 150, 175, 200]#, 225, 250, 275, 300]
		# input_amplitudes = [ia * 1000 for ia in input_amplitudes]
		# frequencies = [0.25, 0.5, 0.75, 1]
		frequencies = [0.01, 0.41, .81, 1.21]
		frequencies = [0.002, 0.008, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]#, 0.6, 0.8, 1.0, 1.2]
		frequencies = [0.1, 0.5]
		# frequencies = [0.01, 0.03, 0.05, 0.07]
		amplitudes = [2, 2]#, 2, 2, 2, 2]#, 1, 1, 1, 1]

		input_spikes_init = []
		target_spikes_init = []
		for i in range(len(input_amplitudes)):
			input_spikes_init.append(input_amplitudes[i] * torch.ones((steps, 1, 1, input_size)))
			# input_spikes_init.append(input_amplitudes[i] * ut.generate_sinusoid(spike_r, frequencies[i], sim_time, dt, amplitude = amplitudes[i]).reshape(steps, 1, 1, input_size))
			target_spikes_init.append(ut.generate_sinusoid(spike_r, frequencies[i], sim_time, dt, amplitude = amplitudes[i]).reshape(steps, 1, 1, 1))
			# target_spikes_init.append(0.02 * input_amplitudes[i] * torch.ones((steps, 1, 1, 1)))
			# input_spikes_init.append(ut.generate_sinusoid(spike_r, frequencies[i], sim_time, dt, amplitude = amplitudes[i]).repeat(input_size).reshape(steps, 1, 1, input_size))

		# input_spikes_init = []
		# target_spikes_init = []

		# input_spikes_init.append(torch.zeros((steps, 1, 1, input_size)))
		# target_spikes_init.append(torch.zeros((steps, 1, 1, input_size)))

		# input_spikes_init.append(25000 * torch.ones((steps, 1, 1, input_size)))
		# target_spikes_init.append(ut.generate_sinusoid(spike_r, 1, sim_time, dt, amplitude = 1).reshape(steps, 1, 1, 1))
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
	# model = MyRNN(1, 200, 1)

	# lr = 1
	# criterion = nn.MSELoss()
	# optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	# for epoch in range(100):
	# 	hidden_state = model.init_hidden()
	# 	for freq in [0.00002]:
	# 		input = freq * torch.ones((400,1,1))
	# 		target = torch.arange(start = 0, end = 400, step = 1) * 2 * math.pi
	# 		target = torch.sin(target * (freq)) * 2 + 10
	# 		target = target.reshape((1,1,400))

	# 		outputs = []
	# 		for i in range(len(input)):
	# 			output, hidden_state = model(input[i,:,:], hidden_state)
	# 			outputs.append(output)
	# 		loss = criterion(torch.stack(outputs, -1), target)
	# 		print(loss)
	# 		optimizer.zero_grad()
	# 		loss.backward()
	# 		optimizer.step()
	# plt.plot(target.reshape(400), label="target")
	# plt.plot(outputs, label="output")
	# plt.legend()
	# plt.show()










	train_network = BNN3L(layers, sizes, conns, spike_r = spike_r, delay = delay, dt=dt, batch_size=batch_size)
	train_network.eval()
	for i in [0, 100, 300, 400, 1000]:
		train_network.reset_state(batch_size=1)
		input_spikes =  i * torch.ones(steps, 1, 1, input_size)
		train_network.forward(input_spikes, rnn=rnn)
		# input_spikes = ut.generate_sinusoid(spike_r, 0.01, sim_time, dt, amplitude = 100000000).reshape(steps, 1, 1, input_size)
		train_spikes = train_network.forward(input_spikes, rnn=rnn)
		# train_spikes = torch.stack(train_network._layers["hidden"].voltages, dim=0)
		# train_spikes = torch.mean(train_spikes, dim=-1).detach()
		# plt.plot(np.arange(len(ts) - delay) * dt, train_spikes[delay:].reshape(len(ts)-delay), label = f"{i}")
		plt.plot(np.arange(steps) * dt, torch.stack(train_spikes, dim=0).detach()[-steps:, 0,0,:], label = f"{i}")
	plt.legend()
	plt.show()

	if single:
		train_network.reset_state(batch_size=1)

		init_spikes = train_network.forward(input_spikes, rnn=rnn)
		init_spikes_all = {}
		for name in layers:
			print(len(train_network._layers[name].spikes_all))
			init_spikes_all[name] = torch.stack(train_network._layers[name].spikes_all, dim=0)

		colors = ["palegreen", "yellowgreen", "mediumseagreen", "darkgreen", "yellow"]
		c = -1
		for name in layers:
			c += 1
			for i in range(sizes[name]):
				plt.plot(np.arange(len(init_spikes) - delay) * dt, init_spikes_all[name].detach()[delay:,0,0,i], color = colors[c], label = name if i == 0 else "")
				if not ('average' in layers or '#average' in layers):
					c += 1
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
	for i in range(len(layers)):
		name = layers[i]
		tracker[name + "_activation"] = []

		if name[0] != '#':
			for p in train_params:
				tracker[name + "_" + p] = []
				tracker[name + "_" + p + "_grad"] = []

	for name in conns.keys():
		tracker[name + "_weight"] = []
		tracker[name + "_weight_grad"] = []
		tracker['losses'] = []

	optimizer = optim.Adam(train_network.parameters(), lr=lr)
	# param_list_lr = []
	# for n in train_network._layers.keys():
	# 	if type(train_network._layers[n]) == GLIFR:
	# 		param_list_lr.append({'params': train_network._layers[n].ln_k_m, 'lr': 0.01})
	# 		param_list_lr.append({'params': train_network._layers[n].thresh, 'lr': 0.1})
	# 		param_list_lr.append({'params': train_network._layers[n].asc_amp, 'lr': 0.01})
	# 		param_list_lr.append({'params': train_network._layers[n].asc_r, 'lr': 0.01})
	# for n in train_network._conns.keys():
	# 	param_list_lr.append({'params': train_network._conns[n].parameters(), 'lr':0.1})

	# optimizer = optim.Adam(param_list_lr,lr=lr)

	# for l in train_network._layers.keys():
	# 	for p in train_network._layers[l].parameters():
	# 		p.register_hook(lambda grad: ut.constrain(grad, c=1e2))
	# for l in train_network._conns.keys():
	# 	for p in train_network._conns[l].parameters():
	# 		p.register_hook(lambda grad: ut.constrain(grad, c=1e2))

	## Training loop
	train_spikes_all = {}
	train_spikes = 0

	tsa = {}
	ts = 0

	loss_fn = nn.MSELoss()

	train_network.train()
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
		# 	batch_count = 0
		# 	for batch_ndx, sample in enumerate(dataloader):
		# 		loss = 0
		# 		batch_count += 1
		# 		train_network.reset_state(len(sample))
		# 		with torch.no_grad():
		# 			input_spikes, target_spikes = sample
		# 			a, b, c, d, e = input_spikes.shape
		# 			input_spikes = input_spikes.reshape(b, a, d, e)

		# 			input_spikes_randn = input_spikes + torch.randn_like(input_spikes)

		# 			a, b, c, d, e = target_spikes.shape
		# 			target_spikes = target_spikes.reshape(b, a, d, e)

		# 			target_spikes_randn = target_spikes + torch.randn_like(target_spikes) * 0.001

		# 		ts = train_network.forward(input_spikes_randn, rnn=rnn)
		# 		# print(torch.stack(ts).shape)
		# 		# ts = torch.stack(ts, dim=0)
		# 		for name in layers:
		# 			tsa[name] = torch.stack(train_network._layers[name].spikes_all, dim=0)
		# 		loss = loss + loss_fn(torch.stack(ts[-steps:],dim=0), target_spikes_randn)

		# 		# for j in range(steps):
		# 		# 	loss = loss + ut.avg_norm(ts[j + delay], target_spikes_randn[j,:,:])
		# 		# loss = loss / batch_count
		# 		if batch_ndx < len(dataloader) - 1:
		# 			loss = loss + ut.km_reg(train_network, km_reg_lambda)
		# 			loss.backward()
		# 			optimizer.step()
		# else:
			input_spikes = input_spikes_init[ep%2]
			target_spikes = target_spikes_init[ep%2]
			train_network.forward(input_spikes, rnn=rnn)
			ts = train_network.forward(input_spikes, rnn=rnn)
			for name in layers:
				tsa[name] = torch.stack(train_network._layers[name].spikes_all, dim=0)

			target_spikes_randn = target_spikes + torch.randn_like(target_spikes) * 0.001

			loss = loss + loss_fn(torch.stack(ts[-steps:],dim=0), target_spikes_randn)

			# for j in range(steps):
			# 	loss = loss + ut.avg_norm(ts[j + steps + delay], target_spikes_randn[j,:,:])
		# loss = loss + ut.fft_norm(torch.stack(ts[delay:], dim=0), target_spikes)
		loss = loss + ut.km_reg(train_network, km_reg_lambda) ##TODO: CHANGE back to 50
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

			if name[0] != '#':
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

		for name in conns:#name, layer in target_network._layers.items():
			pre, post = conns[name]
			layer = train_network._conns[name]
			tracker[name + "_weight"].append([layer.weight[j,0].item() + 0.0 for j in range(sizes[post])])

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
				if rnn:#ep < 200:
					with torch.no_grad():
						# layer.asc_k.grad *= 0
						# layer.asc_r.grad *= 0
						# layer.asc_amp.grad *= 0
						# layer.ln_k_m.grad *= 0
						layer.thresh.grad *= 0
				else:
					layer.asc_k.grad *= 0
					# layer.ln_k_m.grad *= 0
				# layer.asc_r.grad *= 0
				# layer.asc_amp.grad *= 0
				for name in conns.keys():
					pre, post = conns[name]
					layer = train_network._conns[name]
					tracker[name + "_weight_grad"].append([layer.weight.grad[j,0] + 0.0 for j in range(sizes[post])])

		optimizer.step()
	train_network.eval()
	# Finsih plotting outputs
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
					plt.savefig(fig_name + "_outputs_" + date)
					plt.xlim(0,10)
					plt.savefig(fig_name + "_outputs_zoom_" + date)
					plt.xlim(30,40)
					plt.savefig(fig_name + "_outputs_zoom_end_" + date)
					plt.close()

	if sussillo or bellec:
		colors = ["orangered", "darkkhaki", "yellowgreen", "cyan", "deepskyblue", "crimson", "magenta", "lawngreen", "darkviolet", "sienna"]
		colors = ["sienna", "peru", "peachpuff", "salmon", "red", "darkorange", "purple", "fuchsia", "plum", "darkorchid", "slateblue", "mediumblue", "cornflowerblue", "skyblue", "aqua", "aquamarine", "springgreen", "green"]
		c = -1
		for j in range(len(input_amplitudes)):
			c += 1
			train_network.reset_state()
			with torch.no_grad():
				input_spikes = input_spikes_init[j]#torch.zeros((steps, 1, 1, input_size))
				target_spikes = target_spikes_init[j]
				train_network.forward(input_spikes, rnn=rnn)
				train_spikes = train_network.forward(input_spikes, rnn=rnn)
				# for name in layers:
				# 	train_spikes_all[name] = torch.stack(train_network._layers[name].spikes_all, dim=0)
				train_spikes = torch.stack(train_spikes, dim=0)
				plt.plot(np.arange(steps) * dt, torch.mean(train_spikes.detach()[-steps:, 0,0,:],-1), color = colors[c], label = f"predicted {frequencies[j]} Hz")
				plt.plot(np.arange(len(target_spikes)) * dt, target_spikes.detach()[:,0,0,0], '--', color = colors[c])
	plt.xlabel('time (ms)')
	plt.ylabel('spiking output (1/ms)')
	plt.legend()
	plt.savefig(fig_name + "_outputs_all_" + date)
	plt.xlim([0,10])
	plt.savefig(fig_name + "_outputs_all_zoom_left" + date)
	plt.xlim([30,40])
	plt.savefig(fig_name + "_outputs_all_zoom_right" + date)
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
		if name[0] != '#':
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

	n = -1
	for name in conns.keys():
		pre, post = conns[name]
		# if post != "":
		for k in [name + "_weight"]:
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
	plt.savefig(fig_name + "_weights_" + date)
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

	n=-1
	for name in conns.keys():
		pre, post = conns[name]
		# if post != "":
		for k in [name + "_weight_grad"]:
			diff = np.array(tracker[k]) 
			_, l = diff.shape
			n += 1
			for j in range(l):
				plt.plot(range(num_epochs), diff[:,j], color = colors[n], label = k if j == 0 else "")

	plt.axhline(y=0, color='k', linestyle='-')
	plt.legend()
	plt.xlabel("# epoch")
	plt.ylabel("gradient")
	# plt.show()
	plt.savefig(fig_name + "_weight-grads_" + date)
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