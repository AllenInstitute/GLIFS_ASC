import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt
import math

import neurons.utils_glif_new as uts


"""
Defines a single layer of rate-based GLIF neurons with after-spike currents.
"""
class GLIFR(nn.Module):
	r""" Defines rate-based GLIF model.
	param cells_shape: tuple specifying shape of cells (1, 1, number of cells) // technically first dimension represents batch size
	param k_m: inverse time constant (1/ms)
	param R: resistance (GOhm)
	param v_reset: reset voltage (mV)
	param thresh: threshold (mV)
	param spike_r: amplitude/gamma
	param sigma_v: smoothness of spiking function
	param I0: input current (pA)
	param k_syn: synaptic decay factor (ms)
	param asc_amp: jump for after spike currents at spike
	param asc_r: multiplicative factor for after spike currents
	param asc_k: decay factor for after spike currents
	param dt: length of single timestep (ms)
	param delay: number of timesteps for synaptic delay

	asc_amp, asc_r, thresh, log of k_m set as nn.Parameters so learnable
	asc_k initialized to uniform distribution
	"""
	def __init__(self, cells_shape, k_m, R, v_reset, thresh, spike_r, sigma_v, I0, k_syn, asc_amp, asc_r, asc_k, dt, delay, syncurrs_init = 0):
		super().__init__()
		self.cells_shape = cells_shape
		self.delay = delay
		if delay == None:
			self.delay = -1

		self.asc_init = [0,0]
		s_hat = spike_r / 2
		for i in range(len(asc_amp)):
			self.asc_init[i] = asc_amp[i] * s_hat / (asc_k[i] - asc_r[i] * s_hat)

		self.syncurrs_init = syncurrs_init

		# Fixed parameters
		self.register_buffer("R", torch.tensor(R, dtype=torch.float))
		self.register_buffer("v_reset", torch.tensor(v_reset, dtype=torch.float, requires_grad=False))
		self.register_buffer("dt", torch.tensor(dt, dtype=torch.float, requires_grad=False))
		self.register_buffer("spike_r", torch.tensor(spike_r, dtype=torch.float, requires_grad=False))
		self.register_buffer("sigma_v", torch.tensor(sigma_v, dtype=torch.float, requires_grad=False))
		self.register_buffer("I0", torch.tensor(I0, dtype=torch.float, requires_grad=False))
		self.register_buffer("k_syn", torch.tensor(k_syn, dtype=torch.float))
		ln_k_m = math.log(k_m)

		# States
		self.spikes_all = [torch.zeros(*cells_shape, dtype=torch.float)]
		self.register_buffer("voltage", torch.empty(*cells_shape, dtype=torch.float))
		self.register_buffer("SYNcurrents", torch.empty(cells_shape, dtype=torch.float, requires_grad=False))
		self.register_buffer("AScurrents", torch.empty((len(asc_amp),)+cells_shape, dtype=torch.float, requires_grad=False))

		# Learnable Parameters
		ln_k_m = math.log(k_m)
		self.thresh = Parameter(thresh * torch.ones(*cells_shape, dtype=torch.float) + 0 * torch.randn(*cells_shape, dtype=torch.float), requires_grad=True)
		self.ln_k_m = Parameter(ln_k_m * torch.ones(*cells_shape, dtype=torch.float), requires_grad=True)
		# nn.init.uniform_(self.ln_k_m, -5, 0)
		self.asc_amp = Parameter(torch.tensor(asc_amp).reshape((len(asc_amp), 1, 1, 1)) * torch.ones((len(asc_amp),)+cells_shape, dtype=torch.float) + 0.000001 * torch.randn((len(asc_amp),)+cells_shape, dtype=torch.float), requires_grad=True)
		ln_asc_k = [math.log(k_j) for k_j in asc_k]
		self.asc_k = Parameter(torch.tensor(ln_asc_k).reshape((len(asc_amp), 1, 1, 1)) * torch.ones((len(asc_amp),)+cells_shape, dtype=torch.float) + 0.000001 * torch.randn((len(asc_amp),)+cells_shape, dtype=torch.float), requires_grad=True)
		self.asc_r = Parameter(torch.tensor(asc_r).reshape((len(asc_amp), 1, 1, 1)) * torch.ones((len(asc_amp),)+cells_shape, dtype=torch.float) + 0.000001 *  torch.randn((len(asc_amp),)+cells_shape, dtype=torch.float), requires_grad=True)
		# self.asc_k = Parameter(torch.uniform(ln_asc_k).reshape((len(asc_amp), 1, 1, 1)) * torch.ones((len(asc_amp),)+cells_shape, dtype=torch.float) + 0.000001 * torch.randn((len(asc_amp),)+cells_shape, dtype=torch.float), requires_grad=True)
		nn.init.uniform_(self.asc_k, -5, 3)

		# Update functions
		self.spike_update = uts.exponential_spiking
		self.voltage_update = uts.voltage_update
		self.AScurrents_update = uts.AScurrents_update
		self.SYNcurrents_update = uts.SYNcurrents_update_current

		# Initialize
		self.reset_state()

	def fold(self, x):
		r"""Fold incoming spike train by summing last dimension."""
		if isinstance(x, (list, tuple)):
			x = torch.cat(x, dim=-1)

		return x.sum(-1)

	def reset_state(self, batch_size=1):
		"""
		Resets voltage, spikes, after spike currents, and synaptic currents to 0.
		param batch_size: current batch size (initalizes neuron layer shape to accomodate for that)
		"""
		with torch.no_grad():
			self.voltage.fill_(0.0)#(0.0)
			self.spikes_all = [torch.zeros(*self.cells_shape, dtype=torch.float)]
			self.voltages = [torch.zeros(*self.cells_shape, dtype=torch.float)]
			for i in range(self.delay - 1):
				self.spikes_all.append(torch.zeros(*self.cells_shape, dtype=torch.float))
				self.voltages.append(torch.zeros(*self.cells_shape, dtype=torch.float))
			self.AScurrents.fill_(0.0)
			self.SYNcurrents.fill_(0.0)
			self.AScurrents = self.AScurrents.detach()
			self.SYNcurrents = self.SYNcurrents.detach()
			self.voltage = self.voltage.detach()

	def update_spikes(self, rnn=False):
		"""
		Updates firing rates based on current voltage.
		"""
		return self.spike_update(
			self.voltage, 
			self.thresh, 
			self.spike_r, 
			self.sigma_v, 
			self.dt,
			rnn
			)
		# return spikes#.clone() ## CLONE TODO

	def update_AScurrents(self, rnn=False):
		"""
		Update after-spike currents based on previous timestep spikes.
		"""
		self.AScurrents = self.AScurrents_update(
			self.AScurrents,
			self.spikes_all[-1],
			self.asc_amp,
			torch.exp(self.asc_k),
			self.asc_r,
			self.dt,
			rnn
			)

	def update_SYNcurrents(self, incoming, rnn=False):
		"""
		Updates synaptic currents based on incoming firing rates.
		"""
		self.SYNcurrents = self.SYNcurrents_update(
			self.SYNcurrents,
			incoming,
			self.k_syn,
			self.dt,
			rnn
			)
		# print(self.SYNcurrents)

	def update_voltage(self, rnn):
		"""
		Updates voltage based on currents and previous firing rates.
		"""
		self.voltage = self.voltage_update(
			self.voltage,
			self.v_reset,
			torch.exp(self.ln_k_m),#torch.sigmoid(self.ln_k_m),#torch.exp(self.ln_k_m),# torch.sigmoid(self.ln_k_m),#-torch.log((1 - self.ln_k_m) / self.ln_k_m),#torch.exp(self.ln_k_m),
			self.R,
			self.I0,
			self.AScurrents,
			self.SYNcurrents,
			self.spikes_all[-1],
			self.dt,
			rnn
			)
		# print(self.voltage)
		# print(self.voltage)

	def forward(self, incoming, rnn=False):
		"""
		Updates states based on incoming firing rates (size: (batch_size, 1, number_cells))
		Returns firing rates based on delay
		"""
		# incoming = self.fold(incoming)
		self.update_AScurrents(rnn)
		self.update_SYNcurrents(incoming, rnn)
		
		self.update_voltage(rnn)
		
		spikes = self.update_spikes(rnn)
		self.spikes_all.append(spikes)
		self.voltages.append(self.voltage)
		return self.spikes_all[-self.delay]#.clone() #TODO

class RNNC(nn.Module):
	def __init__(self, cells_shape, in_size, delay, dt, spike_r = 1, sigma_v = 1):
		super().__init__()
		self.cells_shape = cells_shape
		self.delay = delay
		if delay == None:
			self.delay = 1
		self.linear = nn.Linear(in_features=cells_shape[2] + in_size, out_features=cells_shape[2])
		with torch.no_grad():
			self.linear.bias *= 0# nn.init.constant_(self.linear.weight,2)
		self.register_buffer("voltage", torch.empty(*cells_shape, dtype=torch.float)) # the hidden stated
		self.thresh = 0
		self.spike_r = spike_r
		self.sigma_v = sigma_v
		self.dt = dt

		self.spike_update = uts.exponential_spiking
		self.reset_state()
	
	def update_spikes(self):
		# print(self.voltage)
		return self.spike_update(
			self.voltage, 
			self.thresh, 
			self.spike_r, 
			self.sigma_v, 
			self.dt,
			)

	def update_voltage(self, x):
		# print(x.shape)
		# print(self.spikes_all[-self.delay].shape)
		b, _, _ = x.shape
		prev_spikes = self.spikes_all[-self.delay]
		if prev_spikes.shape[0] == 1:
			prev_spikes = prev_spikes.repeat(b,1,1)
		# print(prev_spikes.shape)
		# print(f"x:{torch.mean(x)}")
		input_signal = torch.cat((x, prev_spikes), dim=-1)
		# print(f"is:{torch.mean(input_signal)}")# print(input_signal.shape)
		self.voltage = self.linear(input_signal)

	def reset_state(self, batch_size = 1):
		r""" Resets voltage, spikes, AScurrents, and SYNcurrents"""
		a, b, c  = self.cells_shape
		self.cells_shape = (batch_size, b, c)
		with torch.no_grad():
			self.voltage.fill_(0.0)
			self.spikes_all = [torch.zeros(*self.cells_shape, dtype=torch.float)]
			for i in range(self.delay - 1):
				self.spikes_all.append(torch.zeros(*self.cells_shape, dtype=torch.float))

	def forward(self, incoming):
		b, _, _ = incoming.shape
		self.update_voltage(incoming)
		# print(torch.mean(self.voltage))
		spikes = self.update_spikes()
		self.spikes_all.append(spikes)
		return self.spikes_all[-self.delay]

# class RNNC(nn.Module):
# 	r""" Defines RNN cell.
# 	param cells_shape: tuple specifying shape of cells
# 	param hidden_size: number of hidden states to maintain
# 	param delay: number of timesteps for delay
# 	"""
# 	def __init__(self, batch_size, input_size, hidden_size, delay):
# 		super().__init__()
# 		self.cells_shape = (batch_size, 1, hidden_size)
# 		self.input_size = input_size
# 		self.delay = delay
# 		if delay == None:
# 			self.delay = 1

# 		# States
# 		# print(cells_shape)
# 		self.spikes_all = [torch.zeros(*self.cells_shape, dtype=torch.float)]
# 		# self.register_buffer("spikes", torch.empty(*cells_shape, dtype=torch.float))

# 		self.cell = nn.RNNCell(input_size = self.input_size, hidden_size = hidden_size)

# 		# Initialize
# 		self.reset_state()

# 	def fold(self, x):
# 		r"""Fold incoming spike train by summing last dimension."""
# 		if isinstance(x, (list, tuple)):
# 			x = torch.cat(x, dim=-1)

# 		return x.sum(-1)

# 	def reset_state(self):
# 		r""" Resets voltage, spikes, AScurrents, and SYNcurrents"""
# 		with torch.no_grad():
# 			self.spikes_all = [torch.zeros(*self.cells_shape, dtype=torch.float)]
# 			for i in range(self.delay - 1):
# 				self.spikes_all.append(torch.zeros(*self.cells_shape, dtype=torch.float))
# 		# self.cell.init_hidden(self.cells_shape[0])
# 		## TODO: detach?

# 	def update_spikes(self, incoming):
# 		# print(self.spikes_all[-1].shape)
# 		# print(incoming.shape)
# 		return self.cell.forward(incoming, torch.squeeze(self.spikes_all[-self.delay], 1))
# 		# return spikes#.clone() ## CLONE TODO

# 	def forward(self, incoming):
# 		# print(f"right here: {incoming.shape}")
# 		incoming = torch.squeeze(incoming, 1)
# 		# incoming = self.fold(incoming)
# 		# print(f"after fold: {incoming.shape}")
# 		spikes = self.update_spikes(incoming)
# 		# print(f"spikes_shape: {spikes.shape}")
# 		self.spikes_all.append(torch.unsqueeze(spikes, 1))
# 		# print(len(self.spikes_all))
# 		return self.spikes_all[-self.delay]#.clone() #TODO



# class Input(nn.Module):
# 	r""" Regular input neuron

# 	:param cells_shape: shape of neurons
# 	:param dt: length of one timestep
# 	"""
# 	def __init__(self, cells_shape, dt):
# 		super().__init__()
# 		self.cells_shape = cells_shape

# 		self.register_buffer("dt", torch.tensor(dt, dtype=torch.float))
# 		self.register_buffer("spikes", torch.zeros(*cells_shape, dtype=torch.float))

# 		self.reset_state()

# 	def reset_state(self):
# 		self.spikes.fill_(0.0)
# 		# self.spikes = self.spikes.detach()

# 	def forward(self, x):
# 		r""" Propagates spikes.
# 		

class Placeholder(nn.Module):
	r""" Defines rate-based GLIF model.
	param cells_shape: tuple specifying shape of cells
	param k_m: inverse time constant
	param R: resistance (GOhm)
	param v_reset: reset voltage (mV)
	param thresh: threshold (mV)
	param spike_r: amplitude/gamma
	param sigma_v: smoothness of spiking function
	param I0: input current
	param k_syn: synaptic decay factor
	param asc_amp: jump for after spike currents at spike
	param asc_r: multiplicative factor for after spike currents
	param asc_k: decay factor for after spike currents
	param dt: length of single timestep (ms)
	"""
	def __init__(self, cells_shape, delay):
		super().__init__()
		self.cells_shape = cells_shape
		self.delay = delay

		# Learnable Parameters
		self.ln_k_m = 0
		self.thresh = 0

		self.reset_state(2)

	def reset_state(self, batch_size=2):
		r""" Resets voltage, spikes, AScurrents, and SYNcurrents"""
		# a, b, c  = self.cells_shape

		# self.cells_shape = (batch_size, b, c)
		with torch.no_grad():
			self.spikes_all = [torch.zeros(*self.cells_shape, dtype=torch.float)]
			for i in range(self.delay - 1):
				self.spikes_all.append(torch.zeros(*self.cells_shape, dtype=torch.float))
			
	def forward(self, incoming, rnn=False):
		self.spikes_all.append(incoming)
		return self.spikes_all[-self.delay]#.clone() #TODO

class PlaceholderWt(nn.Module):
	r""" Defines rate-based GLIF model.
	param cells_shape: tuple specifying shape of cells
	param k_m: inverse time constant
	param R: resistance (GOhm)
	param v_reset: reset voltage (mV)
	param thresh: threshold (mV)
	param spike_r: amplitude/gamma
	param sigma_v: smoothness of spiking function
	param I0: input current
	param k_syn: synaptic decay factor
	param asc_amp: jump for after spike currents at spike
	param asc_r: multiplicative factor for after spike currents
	param asc_k: decay factor for after spike currents
	param dt: length of single timestep (ms)
	"""
	def __init__(self, cells_shape, in_size, delay):
		super().__init__()
		self.cells_shape = cells_shape
		_,_,c = cells_shape
		self.delay = delay

		self.linear = nn.Linear(in_features=in_size, out_features=c)
		with torch.no_grad():
			self.linear.bias *= 0

		# Learnable Parameters
		self.ln_k_m = 0
		self.thresh = 0

		self.reset_state(1)

	def reset_state(self, batch_size=1):
		r""" Resets voltage, spikes, AScurrents, and SYNcurrents"""
		a, b, c  = self.cells_shape

		self.cells_shape = (batch_size, b, c)
		with torch.no_grad():
			self.spikes_all = [torch.zeros(*self.cells_shape, dtype=torch.float)]
			for i in range(self.delay - 1):
				self.spikes_all.append(torch.zeros(*self.cells_shape, dtype=torch.float))
			
	def forward(self, incoming, rnn=False):
		self.spikes_all.append(self.linear(incoming))
		return self.spikes_all[-self.delay]#.clone() #TODO