import math

import torch
import torch.nn as nn

from neurons.glif_new import GLIFR

class RBNN(nn.Module):
	"""
	Defines a recurrent biological neural network using GLIFR neurons.
	Contains single recurrent layer receiving weighted inputs and outputting weighted outputs.

	Parameters
	----------
	in_size : int
		number of inputs
	hid_size : int
		number of neurons in hidden layer
	out_size : int
		number of outputs
	"""
	def __init__(self, in_size, hid_size, out_size, dt, delay, k_m = 0.02, R = 0.1):
		super().__init__()

		self.input_linear = nn.Linear(in_features = in_size, out_features = hid_size, bias = True)
		self.rec_linear = nn.Linear(in_features = hid_size, out_features = hid_size, bias = True)
		self.output_linear = nn.Linear(in_features = hid_size, out_features = out_size, bias = True)
		self.neuron_layer = GLIFR((1,1,hid_size), k_m = k_m, R = R, dt = dt, delay = delay)

		self.in_size = in_size
		self.hid_size = hid_size
		self.out_size = out_size
		self.dt = dt
		self.delay = delay

		# Initialize linear layers
		with torch.no_grad():
			nn.init.constant_(self.input_linear.bias, 10)
			nn.init.constant_(self.output_linear.bias, 10)

			wt_mean = 1 / ((dt ** 2) * self.hid_size * R * k_m) # for whole layer sum
			wt_var_in = 1 / ((dt ** 2) + self.hid_size * self.in_size * R * k_m)
			wt_var_rec = 1 / ((dt ** 2) + self.hid_size * self.hid_size * R * k_m)

			range_wt_in = math.sqrt(12 * wt_var_in)
			range_wt_rec = math.sqrt(12 * wt_var_rec)

			min_wt_in = wt_mean - (range_wt_in / 2)
			max_wt_in = wt_mean + (range_wt_in / 2)

			min_wt_rec = wt_mean - (range_wt_rec / 2)
			max_wt_rec = wt_mean + (range_wt_rec / 2)
			
			nn.init.uniform_(self.input_linear.weight, min_wt_in, max_wt_in)
			nn.init.uniform_(self.rec_linear.weight, min_wt_rec, max_wt_rec)

	def forward(self,input):
		"""
		Propagates input through network.

		Parameters
		----------
		input : Tensor(batch_size, nsteps, 1, in_size)
			input signal to be input over time
		"""
		# print(input.shape)
		_, nsteps, _, in_size = input.shape
		assert(in_size == self.in_size), f"input has {in_size} size but network accepts {self.in_size} inputs"

		# outputs = []
		outputs = torch.empty((self.batch_size, nsteps, 1, self.out_size))
		
		for step in range(nsteps):
			x = input[:, step, :, :]
			x = self.input_linear(x)
			x = x + self.rec_linear(self.neuron_layer.history[-self.delay])
			x = self.neuron_layer(x)
			x = self.output_linear(x)
			# print(x.shape)
			outputs[:, step, :, :] = x
			# outputs.append(x)
		return outputs
	
	def reset_state(self, batch_size = 1):
		"""
		Resets states (voltage and outputs) of network.
		"""
		self.batch_size = batch_size
		self.neuron_layer.reset_state(batch_size)