from collections import OrderedDict
import math
import numpy as np

import torch
import torch.nn as nn

from neurons.glif_new import GLIFR, RNNC, Placeholder, PlaceholderWt
from neurons.synapse_new import Linear


"""
This file contains several classes to facilitate defining neural network classes
"""

class BNN3L(nn.Module):
	r""" Defines a neural network using rate based GLIF-ASC neurons with given architecture
	param layers: list of names of layers in order of propagation
	param sizes: dict of layer names to sizes
	param conns: dict of connection names to tuples specifying named connections
	param dt: duration of timestep (ms)

	Initializes weights based on Xavier-inspired method.
	"""
	def __init__(self, layers, sizes, conns, dt, spike_r = 20, k_m = .2, R = .1, thresh = 0, delay = None, batch_size = 1):
		super().__init__()
		self.layers = layers
		self._conns = nn.ModuleDict({})#OrderedDict()
		self._layers = nn.ModuleDict({})#OrderedDict()
		self._prevs = {k: [] for k in sizes.keys()}
		self.delay = delay
		self.sizes = sizes
		self.dt = dt

		# self.sizes[""] = sizes[layers[-1]]

		v_reset = 0
		sigma_v = 50
		I0 = 700
		k_syn = 1
		asc_amp = (-9.18, -198.94)
		asc_r = (1.0, 1.0)
		asc_k = (0.003, 0.1)
		# asc_amp = (-1, -20)#(-9.18, -198.94)
		# asc_r = (1.0, 1.0)
		asc_k = (0.5,1)#(0.003, 0.1)

		#positive r with longer k and longterm negative amplitude
		#negative r (saturating) with shorter k and positive amplitude
		asc_amp = (-1, 1)
		asc_k = (3, 30)
		# asc_k = (0.001, 0.011)
		asc_r = (1.0, -1.0)

		self.dropout = nn.Dropout(p=0.2)

		# exp_s = 1

		Rs = {}
		k_ms = {}
		wts = {}
		var_wts = {}

		for k in layers:
			Rs[k] = R + (1 / sizes[k]) #TODO: makes sense?
			k_ms[k] = k_m
		for k in conns.keys():
			pre, post = conns[k]
			# wts[k] = 1 / ((dt ** 2) * sizes[post] * sizes[pre] * Rs[post] * k_ms[post])
			wts[k] = 1 / ((dt ** 2) * sizes[post] * Rs[post] * k_ms[post]) # for whole layer sum

			var_wts[k] = 1 / ((dt** 2) * sizes[post] * sizes[pre] * R * k_m)
			print(f"{k}: exp = {wts[k]}, var = {var_wts[k]}")

		for k in conns.keys():
			pre, post = conns[k]

			if pre[0] == '#' or post[0] == '#':
				biasBool = True
			else:
				biasBool = False

			linear = nn.Linear(sizes[pre], sizes[post], bias=biasBool)

			with torch.no_grad():
				if (pre[0] == '#' or post[0] == '#') and sizes[pre] == sizes[post]:
					nn.init.eye_(linear.weight)
				else:
					# Uniform initialization
					range_wts = math.sqrt(12 * (var_wts[k]))
					min_wt = wts[k] - (range_wts / 2)
					max_wt = wts[k] + (range_wts / 2)
					if (pre[0] != '#' and post[0] != '#'):
						nn.init.uniform_(linear.weight, min_wt, max_wt)
					# TODO: explore normal dist too
			self._conns[k] = linear

			self._prevs[post].append((pre, k))

		for i in range(len(layers)):#k in sizes.keys():
			k = layers[i]

			if k[0] != '#':
				self._layers[k] = GLIFR(
					cells_shape = (batch_size, 1, sizes[k]),
					k_m = k_ms[k],
					R = Rs[k],
					v_reset = v_reset,
					thresh = thresh,
					spike_r = spike_r,
					sigma_v = sigma_v,
					I0 = I0,
					k_syn = k_syn,
					asc_amp = asc_amp,
					asc_r = asc_r,
					asc_k = asc_k,
					dt = dt,
					delay = delay
				)
			else:
				self._layers[k] = Placeholder(
					cells_shape = (batch_size, 1, sizes[k]),
					delay = delay
				)
		self.wts = wts

	def forward(self, inputs, rnn=False):
		steps = inputs.shape[0]
		# self.spikes_all = [torch.zeros(*self._layers[self.layers[-1]].cells_shape, dtype=torch.float)]
		# for i in range(self.delay - 1):
		# 	self.spikes_all.append(torch.zeros(*self._layers[self.layers[-1]].cells_shape, dtype=torch.float))
		print(steps)
		for step in range(steps):
			# print(f"ii: {inputs.shape}")
			x = inputs[step, :, :]
			# x = self.input_linear(x)
			# x = inputs[:, step, :, :]
			for i in range(len(self.layers)):
				post = self.layers[i]
				# print(post)
				if len(self._prevs[post]) == 0: # eg, input layer
					x = self._layers[post].forward(x, rnn)
				else:
					y = torch.zeros(self._layers[post].spikes_all[-1].shape)
					n_in = 0
					for pre, conn in self._prevs[post]:
						curr_delay = self._layers[pre].delay
						if post[0] == '#':
							curr_delay = 1
						n_in = n_in + 1
						y = y + self._conns[conn].forward(self._layers[pre].spikes_all[-curr_delay])
					y = y / n_in
					x = self._layers[post].forward(y, rnn)
					# if self.sizes[post] > 100:
					# 	x = self.dropout(x)
			# x = self.output_linear.forward(x)		
			# self.spikes_all.append(x)
		return self._layers[self.layers[-1]].spikes_all#self.spikes_all #self._layers[self.layers[-1]].spikes_all

	def reset_state(self, batch_size=1):
		r""" Resets state of all submodules."""
		for module in self.modules():
			if isinstance(module, GLIFR) or isinstance(module, RNNC) or isinstance(module, Placeholder) or isinstance(module, PlaceholderWt):
				module.reset_state(batch_size)
			# elif not isinstance(module, BNN3L) and not isinstance(module, nn.ModuleDict) and not isinstance(module, nn.Linear) and not isinstance(module, nn.BatchNorm2d) and not isinstance(module, nn.Dropout):
			# 	module.reset_state()

class RNN(nn.Module):
	"""
	Defines a RNN based on given architecture.
	"""
	def __init__(self, layers, sizes, conns, dt, spike_r = 20, k_m = .2, R = .1, thresh = 0, delay = None, batch_size = 1):
		super().__init__()
		self.layers = layers
		# self._conns = nn.ModuleDict({})#OrderedDict()
		self._layers = nn.ModuleDict({})#OrderedDict()
		self._prevs = {k: [] for k in sizes.keys()}
		self.delay = delay
		self.sizes = sizes
		self.dt = dt

		for k in conns.keys():
			pre, post = conns[k]
			self._prevs[post].append(pre)

		for i in range(len(layers)):#k in sizes.keys():
			k = layers[i]
			in_size = sum([self.sizes[pre] for pre in self._prevs[k]])

			if k[0] != '#':
				self._layers[k] = RNNC(
					cells_shape = (1, 1, sizes[k]),
					in_size = in_size,
					delay = self.delay,
					dt = dt
				)
			elif k[1] != '#':
				self._layers[k] = Placeholder(
					cells_shape = (1, 1, sizes[k]),
					# in_size = in_size,
					delay = self.delay
				)
			else:
				self._layers[k] = PlaceholderWt(
					cells_shape = (1, 1, sizes[k]),
					in_size = in_size,
					delay = self.delay
				)

	def forward(self, inputs):
		steps = inputs.shape[0]
		for step in range(steps):
			x = inputs[step, :, :]
			for i in range(len(self.layers)):
				post = self.layers[i]
				if type(self._layers[post]) == Placeholder:
					x = self._layers[post].forward(x)
				else:
					# y = torch.zeros(self._layers[post].spikes_all[-1].shape)
					y = torch.cat(tuple([self._layers[pre].spikes_all[-self.delay] for pre in self._prevs[post]]), dim=-1)
					
					x = self._layers[post].forward(y)
		return self._layers[self.layers[-1]].spikes_all#self.spikes_all #self._layers[self.layers[-1]].spikes_all

	def reset_state(self, batch_size=1):
		r""" Resets state of all submodules."""
		for module in self.modules():
			if isinstance(module, RNNC) or isinstance(module, Placeholder) or isinstance(module, PlaceholderWt):
				module.reset_state(batch_size)

			# if not isinstance(module, RNN) and not isinstance(module, nn.ModuleDict) and not isinstance(module, nn.Linear) and not isinstance(module, nn.BatchNorm2d) and not isinstance(module, nn.Dropout):
			# 	module.reset_state()
# class RNN(nn.Module):
# 	r""" Defines a minimal neural network with a single input layer, a recurrent hidden layer, and an output layer
# 	param layers: list of names of layers in order of propagation
# 	param sizes: dict of layer names to sizes
# 	param conns: dict of connection names to tuples specifying named connections
# 	"""
# 	def __init__(self, layers, sizes, conns, dt, spike_r = 20, k_m = .2, R = .1, thresh = 0, delay = None, batch_size = 1):
# 		super().__init__()
# 		self.layers = layers
# 		# self._conns = nn.ModuleDict({})#OrderedDict()
# 		self._layers = nn.ModuleDict({})#OrderedDict()
# 		self._prevs = {k: [] for k in sizes.keys()}
# 		self.delay = delay
# 		self.sizes = sizes
# 		self.dt = dt

# 		for k in conns.keys():
# 			pre, post = conns[k]
# 			self._prevs[post].append(pre)

# 		for i in range(len(layers)):#k in sizes.keys():
# 			k = layers[i]
# 			in_size = sum([self.sizes[pre] for pre in self._prevs[k]])

# 			if k[0] != '#':
# 				self._layers[k] = RNNC(
# 					out_size = sizes[k],
# 					# cells_shape = (1, 1, sizes[k]),
# 					in_size = in_size,
# 					delay = self.delay,
# 					dt = dt
# 				)
# 			elif k[1] != '#':
# 				self._layers[k] = Placeholder(
# 					cells_shape = (1, 1, sizes[k]),
# 					# in_size = in_size,
# 					delay = self.delay
# 				)
# 			else:
# 				self._layers[k] = PlaceholderWt(
# 					cells_shape = (1, 1, sizes[k]),
# 					in_size = in_size,
# 					delay = self.delay
# 				)

# 	def forward(self, inputs):
# 		steps = inputs.shape[0]
# 		for step in range(steps):
# 			x = inputs[step, :, :]
# 			for i in range(len(self.layers)):
# 				post = self.layers[i]
# 				if type(self._layers[post]) == Placeholder:
# 					x = self._layers[post].forward(x)
# 				else:
# 					# y = torch.zeros(self._layers[post].spikes_all[-1].shape)
# 					y = torch.cat(tuple([self._layers[pre].outputs[-self.delay] for pre in self._prevs[post]]), dim=-1)
					
# 					x = self._layers[post].forward(y)
# 		return self._layers[self.layers[-1]].outputs#self.spikes_all #self._layers[self.layers[-1]].spikes_all

# 	def reset_state(self, batch_size=1):
# 		r""" Resets state of all submodules."""
# 		for module in self.modules():
# 			if isinstance(module, RNNC) or isinstance(module, Placeholder) or isinstance(module, PlaceholderWt):
# 				module.reset_state(batch_size)

# 			# if not isinstance(module, RNN) and not isinstance(module, nn.ModuleDict) and not isinstance(module, nn.Linear) and not isinstance(module, nn.BatchNorm2d) and not isinstance(module, nn.Dropout):
# 			# 	module.reset_state()


# class BNN3L(nn.Module):
# 	r""" Defines a minimal neural network with a single input layer, a recurrent hidden layer, and an output layer
# 	param layers: list of names of layers in order of propagation
# 	param sizes: dict of layer names to sizes
# 	param conns: dict of connection names to tuples specifying named connections
# 	"""
# 	def __init__(self, layers, sizes, conns, targ, dt, spike_r = 20, k_m = .2, R = .1, thresh = 0, delay = None, batch_size=1, batch_norm = False):
# 		super().__init__()
# 		self.layers = layers
# 		# self.sizes = sizes
# 		# self.conns = conns
# 		self._conns = nn.ModuleDict({})#OrderedDict()
# 		self._layers = nn.ModuleDict({})#OrderedDict()
# 		# self._batch_norms = nn.ModuleDict({})# For now using batch norm before each layer of glifs
# 		self._prevs = {k: [] for k in sizes.keys()}
# 		self.batch_norm = batch_norm
# 		self.delay = delay

# 		# self.sizes[""] = sizes[layers[-1]]

# 		v_reset = 0
# 		sigma_v = 5
# 		I0 = 700
# 		k_syn = 1
# 		asc_amp = (-9.18, -198.94)
# 		asc_r = (1.0, 1.0)
# 		asc_k = (0.003, 0.1)
# 		# asc_amp = (-1, -20)#(-9.18, -198.94)
# 		# asc_r = (1.0, 1.0)
# 		asc_k = (1,1)#(0.003, 0.1)

# 		exp_s = 1

# 		for k in conns.keys():
# 			pre, post = conns[k]

# 			if pre[0] == '#' or post[0] == '#':
# 				biasBool = True
# 			else:
# 				biasBool = False

# 			linear = nn.Linear(sizes[pre], sizes[post], bias=biasBool)
# 			wt = sigma_v / (sizes[post] * sizes[pre] * (dt ** 2) * (exp_s - ((1 / spike_r) * ((exp_s ** 2)))))
# 			wt = wt / (R * k_m)
# 			print(f"{k}: {wt}")

# 			if pre[0] == '#' or post[0] == '#':
# 				nn.init.eye_(linear.weight)
# 			# with torch.no_grad():
# 			# 	nn.init.uniform_(linear.weight, -wt, wt)
# 			# 	linear.weight += wt
# 			self._conns[k] = linear

# 			# if post != "":
# 			self._prevs[post].append((pre, k))

# 		# self.output_linear = nn.Linear(sizes[layers[-1]], sizes[layers[-1]], bias=True)
# 		# wt = sigma_v / (sizes[layers[-1]] * sizes[layers[-1]] * (dt ** 2) * (exp_s - ((1 / spike_r) * ((1 / sizes[layers[-1]]) + (exp_s ** 2)))))
# 		# wt = wt / (R * k_m)
# 		# with torch.no_grad():
# 		# 	# nn.init.uniform_(self.output_linear.weight, -wt, wt)
# 		# 	self.output_linear.weight *= wt
# 		# print(f"output linear: {wt}")

# 		for i in range(len(layers)):#k in sizes.keys():
# 			k = layers[i]
# 			# if i == 0:
# 			# 	self.input_linear = nn.Linear(sizes[k], sizes[k], bias=True)
# 			# 	wt = sigma_v / (sizes[k] * sizes[k] * (dt ** 2) * (exp_s - ((1 / spike_r) * ((1 / sizes[k]) + (exp_s ** 2)))))
# 			# 	wt = wt / (R * k_m)
# 			# 	# wt = wt
# 			# 	# with torch.no_grad():
# 			# 	# 	nn.init.uniform_(self.input_linear.weight, -wt, wt)
# 			# 	# 	self.input_linear.weight += wt
# 			# 	print(f"input linear: {wt}")
# 			# if self.batch_norm:
# 			# 	self._batch_norms[k] = nn.BatchNorm2d(sizes[k])
# 			if k[0] != '#':
# 				self._layers[k] = GLIFR(
# 					cells_shape = (batch_size, 1, sizes[k]),
# 					k_m = k_m,# / sizes[k],
# 					R = R, #/ sizes[k],
# 					v_reset = v_reset,
# 					thresh = thresh,
# 					spike_r = spike_r,
# 					sigma_v = sigma_v,
# 					I0 = I0,
# 					k_syn = k_syn,
# 					asc_amp = asc_amp,
# 					asc_r = asc_r,
# 					asc_k = asc_k,
# 					dt = dt,
# 					delay = delay
# 				)
# 			else:
# 				self._layers[k] = Placeholder(
# 					cells_shape = (batch_size, 1, sizes[k])
# 				)
# 		# print(self._prevs)

# 	def forward(self, inputs):
# 		steps = inputs.shape[0]
# 		self.spikes_all = [torch.zeros(*self._layers[self.layers[-1]].cells_shape, dtype=torch.float)]
# 		for i in range(self.delay - 1):
# 			self.spikes_all.append(torch.zeros(*self._layers[self.layers[-1]].cells_shape, dtype=torch.float))

# 		for step in range(steps):
# 			# print(f"ii: {inputs.shape}")
# 			x = inputs[step, :, :]
# 			# x = self.input_linear(x)
# 			# x = inputs[:, step, :, :]
# 			for i in range(len(self.layers)):
# 				post = self.layers[i]
# 				# print(post)
# 				if len(self._prevs[post]) == 0: # eg, input layer
# 					x = self._layers[post].forward(x)
# 				else:
# 					y = torch.zeros(self._layers[post].spikes_all[-1].shape)
# 					n_in = 0
# 					for pre, conn in self._prevs[post]:
# 						# print(x.shape)
# 						n_in = n_in + 1
# 						y = y + self._conns[conn].forward(self._layers[pre].spikes_all[-self._layers[pre].delay])
# 					y = y / n_in
# 					# print(f"y:{y.shape}")
# 					if self.batch_norm:
# 						# x = self._layers[post].forward(y)
# 						x = self._layers[post].forward(self._batch_norms[post].forward(y))
# 					else:
# 						x = self._layers[post].forward(y)
# 			# x = self.output_linear.forward(x)		
# 			self.spikes_all.append(x)
# 		return self._layers[self.layers[-1]].spikes_all#self.spikes_all #self._layers[self.layers[-1]].spikes_all

# 	def reset_state(self):
# 		r""" Resets state of all submodules."""
# 		for module in self.modules():
# 			if not isinstance(module, BNN3L) and not isinstance(module, nn.ModuleDict) and not isinstance(module, nn.Linear) and not isinstance(module, nn.BatchNorm2d):
# 				module.reset_state()

class RNN3L(nn.Module):
	r""" Defines a minimal neural network with a single input layer, a recurrent hidden layer, and an output layer
	param layers: list of names of layers in order of propagation
	param sizes: dict of layer names to sizes
	param conns: dict of connection names to tuples specifying named connections
	"""
	def __init__(self, layers, sizes, conns, delay, batch_size=1):
		super().__init__()
		self.layers = layers
		self.sizes = sizes
		self.conns = conns
		self._conns = nn.ModuleDict({})#OrderedDict()
		self._layers = nn.ModuleDict({})#OrderedDict()
		self._prevs = {k: [] for k in sizes.keys()}

		for k in conns.keys():
			print(k)
			pre, post = conns[k]
			linear = nn.Linear(sizes[pre], sizes[post], bias=False)
			self._conns[k] = linear
			self._prevs[post].append((pre, k))

		prev = layers[0]
		for k in sizes.keys():
			self._layers[k] = RNNC(
				batch_size = batch_size,
				input_size = sizes[prev],
				hidden_size = sizes[k],
				delay = delay
			)
			prev = k
		# print(self._prevs)

	def forward(self, inputs):
		steps = inputs.shape[0]
		for step in range(steps):
			x = inputs[step, :, :]
			for i in range(len(self.layers)):
				post = self.layers[i]
				# print(post)
				# if len(self._prevs[post]) == 0: # eg, input layer
					# print(f"going directly to {post}")
				x = self._layers[post].forward(x)
				# else:
				# 	y = torch.zeros(self._layers[post].spikes_all[-1].shape)
				# 	# print(y.shape)
				# 	for pre, conn in self._prevs[post]:
				# 		# print(x.shape)
				# 		y = y + self._conns[conn].forward(self._layers[pre].spikes_all[-self._layers[pre].delay])
				# 	x = self._layers[post].forward(y)			
		return self._layers[self.layers[-1]].spikes_all

	def reset_state(self):
		r""" Resets state of all submodules."""
		for module in self.modules():
			if not isinstance(module, RNN3L) and not isinstance(module, nn.ModuleDict) and not isinstance(module, nn.Linear) and not isinstance(module, nn.RNNCell) and not isinstance(module, nn.GRUCell) and not isinstance(module, nn.LSTMCell):
				module.reset_state()

# class BNN3L(nn.Module):
# 	r""" Defines a minimal neural network with a single input layer, a recurrent hidden layer, and an output layer
# 	param layers: list of names of layers in order of propagation
# 	param sizes: dict of layer names to sizes
# 	param conns: dict of connection names to tuples specifying named connections
# 	"""
# 	def __init__(self, layers, sizes, conns, targ, dt, spike_r = 20, k_m = 0.2, R = 0.1, thresh = 0, delay = None, batch_size=1, batch_norm = False):
# 		super().__init__()
# 		self.layers = layers
# 		self.sizes = sizes
# 		self.conns = conns
# 		self._conns = nn.ModuleDict({})#OrderedDict()
# 		self._layers = nn.ModuleDict({})#OrderedDict()
# 		self._batch_norms = nn.ModuleDict({})# For now using batch norm before each layer of glifs
# 		self._prevs = {k: [] for k in sizes.keys()}
# 		self.batch_norm = batch_norm

# 		v_reset = 0
# 		# R = 1 / sum([sizes[k] for k in sizes.keys()])#9.43
# 		# I0 = 700
# 		# spike_r = 20
# 		sigma_v = 10
# 		I0 = 700
# 		k_syn = 1
# 		asc_amp = (-9.18, -198.94)
# 		asc_r = (1.0, 1.0)
# 		asc_k = (0.003, 0.1)
# 		# k_m = 0.2 # / sum([sizes[k] for k in sizes.keys()])
# 		# R = 0.1 #1 / (k_m * I0 * dt)
# 		# R = 5
# 		# k_m = 0.2
# 		# asc_sum = 209.121076
# 		wt_all = 100
# 		thresh_all = 1
# 		s_hat = spike_r / 2

# 		asc_sum = 0
# 		for i in range(len(asc_amp)):
# 			asc_sum += asc_amp[i] * s_hat / (asc_k[i] - asc_r[i] * s_hat)

# 		# Average spiking output
# 		avg = 0
# 		# print(targ.shape)
# 		with torch.no_grad():
# 			avg = torch.sum(torch.mean(targ, (0)), -1).data 
# 			# print(f"avg: {avg}")

# 		dsdt = 0.9

# 		for k in conns.keys():
# 			pre, post = conns[k]
# 			avg_lyr = avg / sizes[post]
# 			# R_lyr = R #* sizes[post]
# 			# k_m = 1 / (R_lyr * I0 * dt)

# 			wt = 0
# 			with torch.no_grad():
# 				wt = sigma_v * (dsdt / (sizes[post])) * (1 / (R * k_m))
# 				wt = wt / (avg_lyr * (1 - (avg_lyr / spike_r)))
# 				wt = wt / (dt ** 2)
# 				# wt = wt / (sizes[post] ** 2)
# 			# print(f"wt: {wt}")
# 			wt = wt_all
# 			wt = torch.tensor(wt)
# 			# wt = torch.tensor(0.09)

# 			linear = nn.Linear(sizes[pre], sizes[post], bias=False)
# 			# with torch.no_grad():
# 			# 	nn.init.uniform_(linear.weight, wt.item(), wt.item())
# 			self._conns[k] = linear
# 			self._prevs[post].append((pre, k))

# 		for i in range(len(layers)):#k in sizes.keys():
# 			k = layers[i]
# 			n_in = 1 if i == 0 else sizes[layers[i-1]]
# 			avg_lyr = avg / sizes[k]
# 			syncurrs_init = n_in * wt_all * s_hat / k_syn

# 			# wt = 0
# 			# with torch.no_grad():
# 			# 	wt = sigma_v * (dsdt / sizes[k]) * (1 / ((R * k_m))) / (sizes[post] ** 3)
# 			# 	wt = wt / ((dt ** 2) * avg_lyr * (1 - (avg_lyr / spike_r)))

# 			# isyn = avg * wt * dt
# 			# iasc = sum(asc_amp) * avg_lyr * dt
# 			vprime = 0#R * k_m * (I0 + isyn + iasc) * dt

# 			quot = 1 / (sizes[k] * wt_all * (dt ** 2))

# 			# thresh = 0

# 			# with torch.no_grad():
# 			# 	thresh = vprime + sigma_v * math.log((spike_r - avg_lyr) / avg_lyr)
			
# 			# thresh = 1
# 			wt = 1 if i == 0 else wt_all
# 			# thresh = (1 / (k_m + s_hat))
# 			# thresh = thresh * ((R * k_m * (I0 + asc_sum + (n_in * wt * s_hat / k_syn))) + (s_hat * v_reset))
# 			# thresh = thresh_all
# 			# print(f"thresh_{k}: {thresh}")

# 			# R = thresh * (k_m + s_hat) - (s_hat * v_reset)
# 			# R = R / (k_m * ((I0 + asc_sum + (n_in * wt * s_hat / k_syn))))
# 			# print(f"R_{k}: {R}")
# 			# k_m = (quot * (I0 + asc_sum)) + ((0.9 * spike_r) / (2 * (dt ** 2))) - (spike_r / 2)
# 			# print(f"k_m: {k_m}")
# 			# R = quot / k_m
# 			# print(f"R: {R}")
# 			# isyn = avg * wt * dt
# 			# iasc = torch.sum(asc_amp)
# 			# vprime = thresh - sigma_v * math.log((spike_r - avg_lyr) / avg_lyr)

# 			# prod = avg_lyr * (vprime - v_reset) / (I0 + isyn + iasc)
# 			# asc_amp_lyr = ((0.1 / (R * k_m * dt)) - I0 - isyn) / (avg_lyr * dt)
# 			# asc_amp_lyr = (asc_amp_lyr - 100, asc_amp_lyr + 100)
# 			# print(asc_amp_lyr)

# 			if self.batch_norm:
# 				self._batch_norms[k] = nn.BatchNorm2d(sizes[k])
# 			self._layers[k] = GLIFR(
# 				cells_shape = (batch_size, 1, sizes[k]),
# 				k_m = k_m,# / sizes[k],
# 				R = R, #/ sizes[k],
# 				v_reset = v_reset,
# 				thresh = thresh,
# 				spike_r = spike_r,
# 				sigma_v = sigma_v,
# 				I0 = I0,
# 				k_syn = k_syn,
# 				asc_amp = asc_amp,
# 				asc_r = asc_r,
# 				asc_k = asc_k,
# 				dt = dt,
# 				delay = delay,
# 				syncurrs_init = syncurrs_init
# 				# sim_time = sim_time
# 			)
# 		# print(self._prevs)

# 	def forward(self, inputs):
# 		steps = inputs.shape[0]
# 		for step in range(steps):
# 			# print(f"ii: {inputs.shape}")
# 			x = inputs[step, :, :]
# 			# x = inputs[:, step, :, :]
# 			for i in range(len(self.layers)):
# 				post = self.layers[i]
# 				# print(post)
# 				if len(self._prevs[post]) == 0: # eg, input layer
# 					x = self._layers[post].forward(x)
# 				else:
# 					y = torch.zeros(self._layers[post].spikes_all[-1].shape)
# 					n_in = 0
# 					for pre, conn in self._prevs[post]:
# 						# print(x.shape)
# 						n_in = n_in + 1
# 						y = y + self._conns[conn].forward(self._layers[pre].spikes_all[-self._layers[pre].delay])
# 					y = y / n_in
# 					# print(f"y:{y.shape}")
# 					if self.batch_norm:
# 						# x = self._layers[post].forward(y)
# 						x = self._layers[post].forward(self._batch_norms[post].forward(y))
# 					else:
# 						x = self._layers[post].forward(y)			
# 		return self._layers[self.layers[-1]].spikes_all

# 	def reset_state(self):
# 		r""" Resets state of all submodules."""
# 		for module in self.modules():
# 			if not isinstance(module, BNN3L) and not isinstance(module, nn.ModuleDict) and not isinstance(module, nn.Linear) and not isinstance(module, nn.BatchNorm2d):
# 				module.reset_state()

# class RNN3L(nn.Module):
# 	r""" Defines a minimal neural network with a single input layer, a recurrent hidden layer, and an output layer
# 	param layers: list of names of layers in order of propagation
# 	param sizes: dict of layer names to sizes
# 	param conns: dict of connection names to tuples specifying named connections
# 	"""
# 	def __init__(self, layers, sizes, conns, targ, dt, delay = None, batch_size=1):
# 		super().__init__()
# 		self.layers = layers
# 		self.sizes = sizes
# 		self.conns = conns
# 		self._conns = nn.ModuleDict({})#OrderedDict()
# 		self._layers = nn.ModuleDict({})#OrderedDict()
# 		self._prevs = {k: [] for k in sizes.keys()}

# 		v_reset = 0
# 		spike_r = 20
# 		sigma_v = 10
# 		I0 = 700
# 		k_syn = 0.1
# 		k_m = 0.2 # / sum([sizes[k] for k in sizes.keys()])
# 		R = 0.1 #1 / (k_m * I0 * dt)

# 		# Average spiking output
# 		avg = 0
# 		print(targ.shape)
# 		with torch.no_grad():
# 			avg = torch.sum(torch.mean(targ, (0)), -1).data 
# 			print(f"avg: {avg}")

# 		dsdt = 0.9

# 		for k in conns.keys():
# 			print(k)
# 			pre, post = conns[k]
# 			avg_lyr = 10#avg / sizes[post]
# 			# R_lyr = R #* sizes[post]
# 			# k_m = 1 / (R_lyr * I0 * dt)

# 			wt = 0
# 			with torch.no_grad():
# 				wt = sigma_v * (dsdt / sizes[post]) * (1 / ((R * k_m))) 
# 				wt = wt / (avg_lyr * (1 - (avg_lyr / spike_r)))
# 				wt = wt / (sizes[post] ** 2)
# 			print(f"wt: {wt}")

# 			linear = nn.Linear(sizes[pre], sizes[post], bias=False)
# 			# with torch.no_grad():
# 			# 	nn.init.uniform_(linear.weight, wt.item(), wt.item() + 1)
# 			self._conns[k] = linear
# 			self._prevs[post].append((pre, k))

# 		prev = layers[0]
# 		for k in sizes.keys():
# 			avg_lyr = 10#avg / sizes[k]

# 			self._layers[k] = RNNC(
# 				batch_size = batch_size,
# 				input_size = sizes[prev],
# 				# cells_shape = (batch_size, 1, sizes[k]),
# 				hidden_size = sizes[k],
# 				delay = delay
# 				# sim_time = sim_time
# 			)
# 			prev = k
# 		print(self._prevs)

# 	def forward(self, inputs):
# 		steps = inputs.shape[0]
# 		for step in range(steps):
# 			x = inputs[step, :, :]
# 			for i in range(len(self.layers)):
# 				post = self.layers[i]
# 				# print(post)
# 				# if len(self._prevs[post]) == 0: # eg, input layer
# 					# print(f"going directly to {post}")
# 				x = self._layers[post].forward(x)
# 				# else:
# 				# 	y = torch.zeros(self._layers[post].spikes_all[-1].shape)
# 				# 	# print(y.shape)
# 				# 	for pre, conn in self._prevs[post]:
# 				# 		# print(x.shape)
# 				# 		y = y + self._conns[conn].forward(self._layers[pre].spikes_all[-self._layers[pre].delay])
# 				# 	x = self._layers[post].forward(y)			
# 		return self._layers[self.layers[-1]].spikes_all

# 	def reset_state(self):
# 		r""" Resets state of all submodules."""
# 		for module in self.modules():
# 			if not isinstance(module, RNN3L) and not isinstance(module, nn.ModuleDict) and not isinstance(module, nn.Linear) and not isinstance(module, nn.RNNCell) and not isinstance(module, nn.GRUCell) and not isinstance(module, nn.LSTMCell):
# 				module.reset_state()

# class BNN3L(nn.Module):
# 	r""" Defines a minimal neural network with a single input layer, a recurrent hidden layer, and an output layer
# 	param ins: number of neurons in input layer
# 	param his: number of neurons in hidden layer
# 	param bes: number of neurons in layer before last
# 	param ous: number of neurons in output/target layer
# 	"""
# 	def __init__(self, ins, his, bes, ous, targ, dt, batch_size=1):
# 		super().__init__()
# 		R = 1 / 9.43
# 		I0 = 700
# 		sigma_v = 10
# 		spike_r = 20
# 		k_m = 1 / (R * I0 * dt)

# 		# Average spiking output
# 		average = 0
# 		with torch.no_grad():
# 			average = torch.mean(torch.mean(targ, (0)), -1).data * ous
# 			print(f"avg: {average}")

# 		dsdt = 0.9

# 		self.layers = []
# 		self._layers = OrderedDict()

# 		names = ["input", "hidden", "hidden2", "output"]
# 		sizes = [ins, his, bes, ous]
# 		wts = []
# 		for i in range(len(names)):
# 			avg_lyr = average / sizes[i]
# 			thresh = 0
# 			with torch.no_grad():
# 				thresh = sigma_v * math.log((spike_r - avg_lyr) / avg_lyr)

# 			wt = 0
# 			with torch.no_grad():
# 				wt = sigma_v * (dsdt / sizes[i]) * (1 / R * k_m)
# 				wt = wt / (dt * avg_lyr * (1 - (avg_lyr / spike_r)))
# 			wts.append(wt)

# 			self._layers[names[i]] = GLIFR(
# 				cells_shape = (batch_size, 1, sizes[i]),
# 				k_m = k_m,
# 				R = R,
# 				v_reset = 0,
# 				thresh = thresh,
# 				spike_r = 20,
# 				sigma_v = 10,
# 				I0 = 700,
# 				k_syn = 1,
# 				asc_amp = (-9.18, -198.94),
# 				asc_r = (1.0, 1.0),
# 				asc_k = (0.003, 0.1),
# 				dt = dt,
# 				delay = delay
# 				# sim_time = sim_time
# 			)

# 		names = ["input_for", "hidden_rec", "hidden_for", "hidden2_for"]
# 		archi = [(0,1), (1,1), (1,2), (2,3)]

# 		#TODO: Generalize based on archi (which will guide forward and layer formation) and names and sizes!!!!
# 		for i in range(len(names)):
# 			pre, post = archi[i]
# 			linear = nn.Linear(sizes[pre], sizes[post], bias=False)
# 			with torch.no_grad():
# 				nn.init.uniform_(linear.weight, wts[post].item() - 1, wts_post.item() + 1)
# 			self._layers[names[i]] = linear

# 	def forward(self, inputs):
# 		steps = inputs.shape[0]
# 		for step in range(steps):
# 			x = inputs[step, :, :]
# 			x = self._layers["input"].forward(x) 
# 			x = self._layers["input_for"].forward(x) 
# 			x_rec = self._layers["hidden_rec"].forward(self._layers["hidden"].spikes_all[-self._layers["hidden"].delay])
# 			x = self._layers["hidden"].forward(x + x_rec)
# 			x = self._layers["hidden_for"].forward(x)
# 			x = self._layers["hidden2"].forward(x + x_rec)
# 			x = self._layers["hidden2_for"].forward(x)
# 			x = self._layers["output"].forward(x)
# 			# self.network.forward(inputs[step, :, :])
# 		return self._layers["output"].spikes_all

# 	def reset_state(self):
# 		r""" Resets state of all submodules."""
# 		for module in self.network.modules():
# 			if not isinstance(module, BNN3L) and not isinstance(module, nn.Sequential) and not isinstance(module, nn.Linear):
# 				module.reset_state()

class SNNNetwork(nn.Module):
	def __init__(self, architecture, target, sim_time, tau = 5, R = 1 / 9.43, I0 = 700, dt = 0.05, batch_size = 1, delay = None):
		super().__init__()
		target = target.data
		n = len(architecture)
		print(n)

		if delay == None:
			self.delay = 1
		else:
			self.delay = delay

		sigma_v = 10
		spike_r = 20

		# Average spiking output
		average = 0
		with torch.no_grad():
			average = torch.mean(torch.mean(target, (0)), -1).data * architecture[-1]

		dsdt = 0.9

		self.layers = []
		self._layers = OrderedDict()
		for i in range(n):
			# input layer = 0
			# then 1st, 2nd, etc.
			# last layer is n - 2
			name = f"fc{i}"
			# print(name)
			j = i - 1 # "presynaptic" index
			# print(j)

			average_layer = average / architecture[j+1]
			print("average_layer: " + average_layer)

			thresh = 0
			with torch.no_grad():
				thresh = sigma_v * math.log((1 / average_layer) * (spike_r - (average_layer)))

			k_m = 0
			with torch.no_grad():
				k_m = 1 / (R * I0 * dt)

			wt = 0
			with torch.no_grad():
				wt = (1 / (k_m * R)) * sigma_v * (dsdt / architecture[j+1])
				wt = wt / (dt * average_layer * (1 - (average_layer / gamma)))
				# wt = (tau / R) * (sigma_v / dt) * (dsdt / architecture[j+1])
				# wt = wt * (1 / (average * (1 - (1 / spike_r) * (average)))) / (1000)# * architecture[j+1])
			
			# Add a neuron
			if j >= 0:
				linear = nn.Linear(architecture[j], architecture[j+1], bias=False)
				with torch.no_grad():
					nn.init.uniform_(linear.weight, wt.item() - 1, wt.item() + 1)
					# nn.init.constant_(linear.weight, wt.item())
				self.layers.append(linear)
			else:
				linear = None
			# print(linear.weight)
			# with torch.no_grad():
			# 	eg = (linear.weight)
			# 	nn.init.constant_(eg, wt)
			# 	linear.weight.copy_(eg)
			glif = GLIFR(
				cells_shape = (batch_size, 1, architecture[j+1]),
				k_m = k_m,#1 / tau,
				R = R,
				v_reset = 0,
				thresh = thresh,
				spike_r = 20,
				sigma_v = 10,
				I0 = I0,
				k_syn = 1,
				asc_amp = (-9.18, -198.94),
				asc_r = (1.0, 1.0),
				asc_k = (0.003, 0.1),
				dt = dt,
				delay = self.delay
				# sim_time = sim_time
			)
			self.layers.append(glif) # for layer i+1
			self._layers[name] = {'connection': linear, 'neuron': glif}
		print(self._layers)
		self.network = nn.Sequential(*self.layers) ## TODO: set biases false
		self.reset_state()
		
	def forward(self, inputs):
		steps = inputs.shape[0]
		for step in range(steps):
			self.network.forward(inputs[step, :, :])
			# print(self.layers[-1].spikes_all)
		# print(len(self.layers[-1].spikes_all))
		# print(torch.stack(self.layers[-1].spikes_all, dim=0).shape)
		# return torch.stack(self.layers[-1].spikes_all, dim=0)
		return self.layers[-1].spikes_all
		# for step in range(steps):
			# self.network.forward(inputs[step, :, :].clone())

	def reset_state(self):
		r""" Resets state of all submodules."""
		for module in self.network.modules():
			if not isinstance(module, SNNNetwork) and not isinstance(module, nn.Sequential) and not isinstance(module, nn.Linear):
				module.reset_state()

class RSNNNetwork(nn.Module):
	def __init__(self, architecture, target, sim_time, tau = 5, R = 1 / 9.43, dt = 0.05, batch_size = 1, delay = None):
		super().__init__()
		target = target.data
		n = len(architecture)
		print(n)

		if delay == None:
			self.delay = 1
		else:
			self.delay = delay

		sigma_v = 10
		spike_r = 20

		# Average spiking output
		average = 0
		with torch.no_grad():
			average = torch.mean(torch.mean(target, (0)), -1).data
			print(f"avg: {average}")
		dsdt = 0.9

		self.layers = []
		self._layers = OrderedDict()
		for i in range(n):
			# input layer = 0
			# then 1st, 2nd, etc.
			# last layer is n - 2
			name = f"fc{i}"
			print(name)
			j = i - 1
			print(j)
			thresh = 0
			with torch.no_grad():
				thresh = sigma_v * math.log((1 / average) * (spike_r - (average)))

			wt = 0
			with torch.no_grad():
				wt = (tau / R) * (sigma_v / dt) * (dsdt / (architecture[j+1] ** 2))
				wt = wt * (1 / (average * (1 - (1 / spike_r) * (average)))) / (1000)# * architecture[j+1])
			# Add connection
			if j >= 0:
				linear = nn.Linear(architecture[j], architecture[j+1], bias=False)
				with torch.no_grad():
					nn.init.uniform_(linear.weight, wt.item() - 1, wt.item() + 1)
					# nn.init.constant_(linear.weight, wt.item())
				self.layers.append(linear)
			else:
				linear = None

			# Add a neuron
			glif = GLIFR(
				cells_shape = (batch_size, 1, architecture[j+1]),
				k_m = 1 / tau,
				R = R,
				v_reset = 0,
				thresh = thresh,
				spike_r = 20,
				sigma_v = 10,
				I0 = 700,
				k_syn = 1,
				asc_amp = (-9.18, -198.94),
				asc_r = (1.0, 1.0),
				asc_k = (0.003, 0.1),
				dt = dt,
				delay = delay
				# sim_time = sim_time
			)

			self.layers.append(glif) # for layer i+1
			self._layers[name] = {'connection': linear, 'neuron': glif}

			if i == 1:
				# Add recurrent connection
				linear = nn.Linear(architecture[j+1], architecture[j+1], bias=False)
				with torch.no_grad():
					nn.init.uniform_(linear.weight, wt.item() - 1, wt.item() + 1)
					# nn.init.constant_(linear.weight, wt.item())
				self.layers.append(linear)
				self.layers.append(glif) # for layer i+1
				self._layers[name] = {'connection': linear, 'neuron': glif}

		print(self._layers)
		self.network = nn.Sequential(*self.layers) ## TODO: set biases false
		self.reset_state()
		
	def forward(self, inputs):
		steps = inputs.shape[0]
		for step in range(steps):
			self.network.forward(inputs[step, :, :])
			# print(self.layers[-1].spikes_all)
		# print(len(self.layers[-1].spikes_all))
		# print(torch.stack(self.layers[-1].spikes_all, dim=0).shape)
		# return torch.stack(self.layers[-1].spikes_all, dim=0)
		return self.layers[-1].spikes_all
		# for step in range(steps):
			# self.network.forward(inputs[step, :, :].clone())

	def reset_state(self):
		r""" Resets state of all submodules."""
		for module in self.network.modules():
			if not isinstance(module, SNNNetwork) and not isinstance(module, nn.Sequential) and not isinstance(module, nn.Linear):
				module.reset_state()



# class SNNNetwork(nn.Module):
# 	def __init__(self):
# 		super().__init__()
# 		self._layers = OrderedDict()
# 		self._synapses = OrderedDict()

# 	def reset_state(self):
# 		r""" Resets state of all submodules."""
# 		for module in self.modules():
# 			if not isinstance(module, SNNNetwork):
# 				module.reset_state()

# 	def add_layer(self, name, synapse, neuron):
# 		if isinstance(synapse, Linear):
# 			ctype = "linear"
# 		elif synapse == None:
# 			ctype = "none"
# 			self.prev = neuron
# 			self.input = neuron

# 		self._layers[name] = {"synapse": synapse, "neuron": neuron, "type": ctype}
# 		self.final = name
# 		if ctype != "none":
# 			self._synapses[name] = {"synapse": synapse, "presynaptic": self.prev, "postsynaptic": neuron}
# 			self.prev = neuron

# 	def forward(self, inputs): # Forward stores the spikes at each time bin
# 		r""" Runs simulation given inputs.
# 		:param inputs: a Tensor of shape [num_steps, batch_size, *input_shape]
# 		"""
# 		steps = inputs.shape[0]
# 		spikes_all = OrderedDict()
# 		for name, synapse in self._synapses.items():
# 			cells_shape = synapse["postsynaptic"].cells_shape
# 			spikes_all[name] = torch.empty(steps, *cells_shape)

# 		for step in range(steps):
# 			spikes = self.input.forward(inputs.clone()[step, :, :]) ## TODO: slice
# 			for name, layer in self._synapses.items():
# 				presynaptic = layer["presynaptic"]
# 				synapse = layer["synapse"]
# 				postsynaptic = layer["postsynaptic"]

# 				spikes = synapse.forward(presynaptic.spikes)
# 				# print(f"1 {spikes}")
# 				# print(f"synapse: {synapse.spikes.shape}")
# 				spikes = postsynaptic.forward(synapse.spikes)
# 				# print(f"2 {spikes}")
# 				spikes_all[name][step, :] = postsynaptic.spikes

# 		return spikes_all

# 	def forward_once(self, inputs): # Forward_once just propagates for one time bin
# 		r""" Runs simulation given inputs.
# 		:param inputs: a Tensor of shape [num_steps, batch_size, *input_shape]
# 		"""
# 		steps = inputs.shape[0]
# 		spikes_all = OrderedDict()
# 		for name, synapse in self._synapses.items():
# 			cells_shape = synapse["postsynaptic"].cells_shape
# 			spikes_all[name] = torch.empty(*cells_shape)

# 		spikes = self.input.forward(inputs) ## TODO: slice
# 		for name, layer in self._synapses.items():
# 			presynaptic = layer["presynaptic"]
# 			synapse = layer["synapse"]
# 			postsynaptic = layer["postsynaptic"]

# 			spikes = synapse.forward(presynaptic.spikes)

# 			# print(f"1 {spikes}")
# 			# print(f"synapse: {synapse.spikes.shape}")
# 			spikes = postsynaptic.forward(synapse.spikes)
# 			print(spikes)
# 			# print(f"2 {spikes}")
# 			spikes_all[name] = postsynaptic.spikes

# 		return self._synapses[self.final]['postsynaptic'].spikes, spikes_all