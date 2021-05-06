import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Linear(nn.Linear): ####CHECK IF THIS WORKS
	r""" Passes spikes with weights"""
	def __init__(self, in_features, out_features, batch_size, dt, delay, weight):
		super().__init__()
		self.synapse_shape = (batch_size, out_features, in_features)
		self.batch_size = batch_size
		self.out_features = out_features
		self.in_features = in_features

		##TODO: incorporate delay
		self.register_buffer("dt", torch.tensor(dt, dtype=torch.float))
		self.register_buffer("spikes", torch.empty(*self.synapse_shape, dtype=torch.float))
		self.weight = Parameter(weight * torch.ones((out_features, in_features)), requires_grad=True)
		self.reset_state()

	def fold(self, x):
		return x.view(self.batch_size, -1, self.out_features, self.in_features)

	def reset_state(self):
		self.spikes.fill_(0.0)
		self.zero_grad()

	def forward(self, incoming):
		# print(f"weight: {self.weight.shape}")
		# print(f"spikes: {incoming.shape}")
		out = incoming * self.weight * 1000
		self.spikes = incoming * self.weight * 1000#self.fold(out)
		return self.spikes ##.clone() TODO