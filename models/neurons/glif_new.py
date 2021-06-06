import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt
import math

import neurons.utils_glif_new as uts


class BNNC(nn.Module):
	def __init__(self, input_size, hidden_size, bias = True):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_ascs = 2

		# self.batchnorm_voltage = nn.BatchNorm1d(num_features = input_size)
		# self.batchnorm_activation = nn.BatchNorm1d(num_features = hidden_size)
		
		self.weight_iv = Parameter(torch.randn((input_size, hidden_size)))
		# self.I0 = Parameter(700*torch.ones((1, hidden_size), dtype=torch.float))
		self.c_m_inv = 0.02
		# self.weight_hh = Parameter(torch.randn((input_size, hidden_size)))

		# self.bias_ih = Parameter(torch.randn((1, hidden_size)))
		# self.bias_hh = Parameter(torch.randn((1, hidden_size)))

		self.thresh = Parameter(torch.zeros((1, hidden_size), dtype=torch.float), requires_grad=True)
		ln_k_m = math.log(0.01)
		self.ln_k_m = Parameter(ln_k_m * torch.ones((1, hidden_size), dtype=torch.float), requires_grad=True)
		asc_amp = (-1, 1)
		asc_r = (1,-1)
		# self.asc_r = Parameter(0.01 * torch.ones((self.num_ascs, 1, hidden_size), dtype=torch.float), requires_grad=True)
		# self.asc_r = Parameter(0.01 * torch.ones((self.num_ascs, 1, hidden_size), dtype=torch.float), requires_grad=True)

		self.asc_amp = Parameter(torch.tensor(asc_amp).reshape((len(asc_amp), 1, 1)) * torch.ones((len(asc_amp),1,hidden_size), dtype=torch.float) + 0 * torch.randn((len(asc_amp),1,hidden_size), dtype=torch.float), requires_grad=True)
		self.ln_asc_k = Parameter(torch.ones((self.num_ascs, 1, hidden_size), dtype=torch.float), requires_grad=True)
		self.asc_r = Parameter(torch.tensor(asc_r).reshape((len(asc_amp), 1, 1)) * torch.ones((len(asc_amp), 1, hidden_size), dtype=torch.float) + 0 *  torch.randn((len(asc_amp), 1, hidden_size), dtype=torch.float), requires_grad=True)		
		# nn.init.uniform_(self.ln_asc_k, -.2, .3)
		# nn.init.uniform_(self.asc_r, -.2, .3)
		# nn.init.uniform_(self.asc_amp, -.2, .3)

		self.v_reset = 0#Parameter(torch.randn((1, hidden_size), dtype=torch.float))
		self.R = 0.1

		self.sigma_v = 100
		self.gamma = 20
		self.dt = 0.05

		with torch.no_grad():
			# wt_mean = 1 / (self.dt * self.hidden_size) # for whole layer sum
			# wt_var = 1 / (self.dt * self.hidden_size * self.input_size / self.c_m)

			# range_wt = math.sqrt(12 * wt_var)

			# min_wt = wt_mean - (range_wt / 2)
			# max_wt = wt_mean + (range_wt / 2)

			nn.init.uniform_(self.weight_iv, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))
			# nn.init.orthogonal_(self.weight_iv)

			wt_mean = 1 / ((self.dt ** 2) * self.hidden_size * torch.mean(self.R * torch.exp(self.ln_k_m))) # for whole layer sum
			wt_var = 1 / ((self.dt ** 2) + self.hidden_size * torch.mean(self.R * torch.exp(self.ln_k_m)))
			range_wt = math.sqrt(12 * wt_var)
			print(range_wt)

			min_wt = wt_mean - (range_wt / 2)
			max_wt = wt_mean + (range_wt / 2)
			# nn.init.uniform_(self.asc_r, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))
			# nn.init.uniform_(self.asc_amp, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))
			# nn.init.uniform_(self.weight_iv, min_wt, max_wt)
			# nn.init.uniform_(self.weight_hh, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))
			# nn.init.uniform_(self.bias_ih, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))
			# nn.init.uniform_(self.bias_hh, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))
			# self.weight_iv /= self.gamma

	def spike_fn(self, x):
		"""
		Propagates input through spiking activation function.
		
		Parameters
		----------
		x : Tensor(any size)
			input to spiking function
		
		Returns
		-------
		Tensor(same size as x)
			tanh(x)
		"""
		# x = self.batchnorm_voltage(x)
		activation = (x - self.thresh) / self.sigma_v
		# activation = self.batchnorm_activation(activation)
		return torch.sigmoid(activation)
		# return torch.tanh(x - (self.thresh))
	
	def forward(self, x, firing, voltage, ascurrent, syncurrent):
		# 1.5, -0.5 for lnasck
		syncurrent = x @ self.weight_iv
		ascurrent = (ascurrent * self.asc_r + self.asc_amp) * firing + (1 - self.dt * torch.exp(self.ln_asc_k)) * ascurrent
		# ascurrent = ascurrent * 0
		voltage = syncurrent + self.dt * torch.exp(self.ln_k_m) * self.R * torch.sum(ascurrent, dim=0) + (1 - self.dt * torch.exp(self.ln_k_m)) * voltage - firing * (voltage - self.v_reset)
		firing = self.spike_fn(voltage)#x @ self.weight_ih + (1 - self.dt * torch.exp(self.ln_k_m)) * hidden)# + self.bias_ih) #+ hidden @ self.weight_hh + self.bias_hh)
		return firing, voltage, ascurrent, syncurrent

class RNNC(nn.Module): # The true RNNC
	def __init__(self, input_size, hidden_size, bias = True):
		super().__init__()
		self.weight_ih = Parameter(torch.randn((input_size, hidden_size)))
		self.weight_hh = Parameter(torch.randn((hidden_size, hidden_size)))

		self.bias = torch.zeros((1, hidden_size))

		# self.bias_ih = Parameter(torch.randn((1, hidden_size)))
		# self.bias_hh = Parameter(torch.randn((1, hidden_size)))

		with torch.no_grad():
			nn.init.normal_(self.weight_ih, 0, 1 / math.sqrt(hidden_size))
			nn.init.normal_(self.weight_hh, 0, 1 / math.sqrt(hidden_size))
		# 	# nn.init.xavier_uniform_(self.weight_ih)
		# 	# nn.init.xavier_uniform_(self.weight_hh)
		# 	# nn.init.uniform_(self.weight_ih, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))
		# 	# nn.init.uniform_(self.weight_hh, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))
		# 	nn.init.constant_(self.bias_ih, 0)#nn.init.uniform_(self.bias_ih, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))
		# 	nn.init.constant_(self.bias_hh, 0)#nn.init.uniform_(self.bias_hh, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))

	def forward(self, x, hidden):
		hidden = torch.mm(x, self.weight_ih) + torch.mm(hidden, self.weight_hh) + self.bias
		hidden = torch.tanh(hidden)
		# print("")
		# print(torch.mean(self.weight_ih))
		# print(torch.mean(self.weight_hh))
		# # print(f"multiplying {x.shape} and {self.weight_ih.shape} to get mm = {torch.mm(x, self.weight_ih)} or matmul = {torch.matmul(x, self.weight_ih)}")
		# # print(f"multiplying {hidden.shape} and {self.weight_hh.shape} to get mm = {torch.mm(hidden, self.weight_hh)} or matmul = {torch.matmul(hidden, self.weight_hh)}")
		# hidden = torch.mm(x, self.weight_ih) + self.bias_ih + torch.mm(hidden, self.weight_hh) + self.bias_hh#torch.tanh(x @ self.weight_ih + self.bias_ih + hidden @ self.weight_hh + self.bias_hh)
		# # print(torch.mean(hidden))
		# hidden = torch.tanh(hidden)# print(torch.mean(hidden))
		return hidden


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
	def __init__(self, cells_shape, dt, delay, k_m = 0.02, R = 0.1, v_reset = 0, thresh = 0, spike_r = 20, sigma_v = 50, I0 = 700, k_syn = 1, asc_amp = (-1, 1), asc_r = (1,-1), asc_k = None):
		super().__init__()
		self.cells_shape = cells_shape
		self.delay = delay
		if delay == None:
			self.delay = -1

		# self.asc_init = [0,0]
		# s_hat = spike_r / 2
		# for i in range(len(asc_amp)):
		# 	self.asc_init[i] = asc_amp[i] * s_hat / (asc_k[i] - asc_r[i] * s_hat)

		self.num_ascs = len(asc_amp)

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
		self.history = [torch.zeros(*cells_shape, dtype=torch.float)]
		self.register_buffer("voltage", torch.empty(*cells_shape, dtype=torch.float))
		self.register_buffer("SYNcurrents", torch.empty(cells_shape, dtype=torch.float, requires_grad=False))
		self.register_buffer("AScurrents", torch.empty((len(asc_amp),)+cells_shape, dtype=torch.float, requires_grad=False))

		# Learnable Parameters
		ln_k_m = math.log(k_m)
		self.thresh = Parameter(thresh * torch.ones(*cells_shape, dtype=torch.float) + 0 * torch.randn(*cells_shape, dtype=torch.float), requires_grad=True)
		self.ln_k_m = Parameter(ln_k_m * torch.ones(*cells_shape, dtype=torch.float), requires_grad=True)
		# nn.init.uniform_(self.ln_k_m, -5, 0)
		self.asc_amp = Parameter(torch.tensor(asc_amp).reshape((len(asc_amp), 1, 1, 1)) * torch.ones((len(asc_amp),)+cells_shape, dtype=torch.float))# + 0.000001 * torch.randn((len(asc_amp),)+cells_shape, dtype=torch.float), requires_grad=True)
		# ln_asc_k = [math.log(k_j) for k_j in asc_k]
		self.asc_k = Parameter(torch.ones((len(asc_amp),)+cells_shape, dtype=torch.float), requires_grad=True)
		self.asc_r = Parameter(torch.tensor(asc_r).reshape((len(asc_amp), 1, 1, 1)) * torch.ones((len(asc_amp),)+cells_shape, dtype=torch.float))# + 0.000001 *  torch.randn((len(asc_amp),)+cells_shape, dtype=torch.float), requires_grad=True)
		# self.asc_k = Parameter(torch.uniform(ln_asc_k).reshape((len(asc_amp), 1, 1, 1)) * torch.ones((len(asc_amp),)+cells_shape, dtype=torch.float) + 0.000001 * torch.randn((len(asc_amp),)+cells_shape, dtype=torch.float), requires_grad=True)
		# nn.init.uniform_(self.asc_k, -5, 3)

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
		a,b,c = self.cells_shape

		self.cells_shape = (batch_size, b, c)
		# print(self.cells_shape)
		with torch.no_grad():
			self.voltage.fill_(0.0)#(0.0)
			self.history = [torch.zeros(*self.cells_shape, dtype=torch.float)]
			for i in range(self.delay - 1):
				self.history.append(torch.zeros(*self.cells_shape, dtype=torch.float))
			self.voltage = torch.empty(*self.cells_shape, dtype=torch.float)
			self.SYNcurrents = torch.empty(self.cells_shape, dtype=torch.float, requires_grad=False)
			self.AScurrents = torch.empty((self.num_ascs,)+self.cells_shape, dtype=torch.float, requires_grad=False)
			self.AScurrents.fill_(0.0)
			self.SYNcurrents.fill_(0.0)
			self.AScurrents = self.AScurrents.detach()
			self.SYNcurrents = self.SYNcurrents.detach()
			self.voltage = self.voltage.detach()

	def update_spikes(self):
		"""
		Updates firing rates based on current voltage.
		"""
		return self.spike_update(
			self.voltage, 
			self.thresh, 
			self.spike_r, 
			self.sigma_v, 
			self.dt
			)
		# return spikes#.clone() ## CLONE TODO

	def update_AScurrents(self, rnn=False):
		"""
		Update after-spike currents based on previous timestep spikes.
		"""
		self.AScurrents = self.AScurrents_update(
			self.AScurrents,
			self.history[-1],
			self.asc_amp,
			torch.exp(self.asc_k),
			self.asc_r,
			self.dt
			)

	def update_SYNcurrents(self, incoming):
		"""
		Updates synaptic currents based on incoming firing rates.
		"""
		self.SYNcurrents = self.SYNcurrents_update(
			self.SYNcurrents,
			incoming,
			self.k_syn,
			self.dt,
			)
		# print(self.SYNcurrents)

	def update_voltage(self):
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
			self.history[-1],
			self.dt
		)
		# print(self.voltage)
		# print(self.voltage)

	def forward(self, incoming):
		"""
		Updates states based on incoming firing rates (size: (batch_size, 1, number_cells))
		Returns firing rates based on delay
		"""
		# incoming = self.fold(incoming)
		self.update_AScurrents()
		self.update_SYNcurrents(incoming)
		
		self.update_voltage()
		
		spikes = self.update_spikes()

		self.history.append(spikes)
		return self.history[-self.delay]#.clone() #TODO

# class RNNC(nn.Module):
# 	def __init__(self, cells_shape, in_size, delay, dt, spike_r = 1, sigma_v = 1):
# 		super().__init__()
# 		self.cells_shape = cells_shape
# 		self.delay = delay
# 		if delay == None:
# 			self.delay = 1
# 		self.linear = nn.Linear(in_features=cells_shape[2] + in_size, out_features=cells_shape[2])
# 		with torch.no_grad():
# 			self.linear.bias *= 0# nn.init.constant_(self.linear.weight,2)
# 		self.register_buffer("voltage", torch.empty(*cells_shape, dtype=torch.float)) # the hidden stated
# 		self.thresh = 0
# 		self.spike_r = spike_r
# 		self.sigma_v = sigma_v
# 		self.dt = dt

# 		self.spike_update = uts.exponential_spiking
# 		self.reset_state()
	
# 	def update_spikes(self):
# 		# print(self.voltage)
# 		return self.spike_update(
# 			self.voltage, 
# 			self.thresh, 
# 			self.spike_r, 
# 			self.sigma_v, 
# 			self.dt,
# 			)

# 	def update_voltage(self, x):
# 		# print(x.shape)
# 		# print(self.history[-self.delay].shape)
# 		b, _, _ = x.shape
# 		prev_spikes = self.history[-self.delay]
# 		if prev_spikes.shape[0] == 1:
# 			prev_spikes = prev_spikes.repeat(b,1,1)
# 		# print(prev_spikes.shape)
# 		# print(f"x:{torch.mean(x)}")
# 		input_signal = torch.cat((x, prev_spikes), dim=-1)
# 		# print(f"is:{torch.mean(input_signal)}")# print(input_signal.shape)
# 		self.voltage = self.linear(input_signal)

# 	def reset_state(self, batch_size = 1):
# 		r""" Resets voltage, spikes, AScurrents, and SYNcurrents"""
# 		a, b, c  = self.cells_shape
# 		self.cells_shape = (batch_size, b, c)
# 		with torch.no_grad():
# 			self.voltage.fill_(0.0)
# 			self.history = [torch.zeros(*self.cells_shape, dtype=torch.float)]
# 			for i in range(self.delay - 1):
# 				self.history.append(torch.zeros(*self.cells_shape, dtype=torch.float))

# 	def forward(self, incoming):
# 		b, _, _ = incoming.shape
# 		self.update_voltage(incoming)
# 		# print(torch.mean(self.voltage))
# 		spikes = self.update_spikes()
# 		self.history.append(spikes)
# 		return self.history[-self.delay]

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
# 		self.history = [torch.zeros(*self.cells_shape, dtype=torch.float)]
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
# 			self.history = [torch.zeros(*self.cells_shape, dtype=torch.float)]
# 			for i in range(self.delay - 1):
# 				self.history.append(torch.zeros(*self.cells_shape, dtype=torch.float))
# 		# self.cell.init_hidden(self.cells_shape[0])
# 		## TODO: detach?

# 	def update_spikes(self, incoming):
# 		# print(self.history[-1].shape)
# 		# print(incoming.shape)
# 		return self.cell.forward(incoming, torch.squeeze(self.history[-self.delay], 1))
# 		# return spikes#.clone() ## CLONE TODO

# 	def forward(self, incoming):
# 		# print(f"right here: {incoming.shape}")
# 		incoming = torch.squeeze(incoming, 1)
# 		# incoming = self.fold(incoming)
# 		# print(f"after fold: {incoming.shape}")
# 		spikes = self.update_spikes(incoming)
# 		# print(f"spikes_shape: {spikes.shape}")
# 		self.history.append(torch.unsqueeze(spikes, 1))
# 		# print(len(self.history))
# 		return self.history[-self.delay]#.clone() #TODO



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
			self.history = [torch.zeros(*self.cells_shape, dtype=torch.float)]
			for i in range(self.delay - 1):
				self.history.append(torch.zeros(*self.cells_shape, dtype=torch.float))
			
	def forward(self, incoming, rnn=False):
		self.history.append(incoming)
		return self.history[-self.delay]#.clone() #TODO

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
			self.history = [torch.zeros(*self.cells_shape, dtype=torch.float)]
			for i in range(self.delay - 1):
				self.history.append(torch.zeros(*self.cells_shape, dtype=torch.float))
			
	def forward(self, incoming, rnn=False):
		self.history.append(self.linear(incoming))
		return self.history[-self.delay]#.clone() #TODO
