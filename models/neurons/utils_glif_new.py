# This code defines utilities (state updates) needed for the GLIFR

import torch
import torch.nn.functional as F

def exponential_spiking(voltage, thresh, spike_r, sigma_v, dt, rnn=False):
	return spike_r * torch.sigmoid((voltage - thresh) / sigma_v)

def linear_spiking(voltage, thresh, spike_r, sigma_v, dt):
	return F.relu((voltage - thresh) / sigma_v)

def voltage_update(voltage, v_reset, k_m, R, I0, AScurrents, SYNcurrents, curr_firing, dt):
	"""
	Computes voltage at next time point

	Parameters
	----------
	voltage : Tensor(batch_size, 1, N) where N is number of neurons
		voltage of cells
	v_reset : float
		reset voltage that neurons tend towards during spiking
	k_m : Tensor(1, 1, N)
		inverse time constant
	R : float
		resistance of neuron
	I0 : float
		constant input current
	AScurrents : Tensor(num_ascs, batch_size, 1, N)
		after-spike currents across all neurons
	SYNcurrents : Tensor(batch_size, 1, N)
		incoming synaptic currents
	curr_firing : Tensor(batch_size, 1, N)
		current firing rate
	
	Returns
	-------
	Tensor(batch_size, 1, N)
		voltage of cells at next timestep
	"""
	currents_sum = I0 + (SYNcurrents) + torch.sum(AScurrents, dim=0) # (batch_size, 1, N)
	rates = curr_firing * dt
	v_delta = 0
	# v_delta = (-dt * k_m - rates) * voltage + rates * v_reset
	v_delta = v_delta + R * k_m * currents_sum * dt
	voltage = voltage + v_delta
	return voltage

def SYNcurrents_update_current(SYNcurrents, incoming, k_syn, dt):
	# SYNcurrents = SYNcurrents - k_syn * SYNcurrents * dt + (incoming * dt)
	return SYNcurrents - k_syn * SYNcurrents * dt + (incoming * dt)

def AScurrents_update(AScurrents, spikes, asc_amp, asc_k, asc_r, dt, rnn=False):
	# print(spikes.shape)
	# print(((asc_amp + asc_r).reshape((2,1,1,1))*AScurrents).shape)
	# print((torch.matmul((asc_amp + asc_r).reshape((2,1,1,1))*AScurrents, spikes)).shape)
	# return ((asc_amp + asc_r).reshape((2,1,1,1))*AScurrents * spikes) * dt + AScurrents * (1 - asc_k.reshape((2,1,1,1)) * dt)
	AScurrents_temp = AScurrents.clone()
	# print((asc_r).shape)
	return spikes * dt * (asc_amp + AScurrents * asc_r) + AScurrents * (1 - asc_k * dt)

	for i in range(len(asc_amp)):
		# print(spikes.shape)
		# print(AScurrents[i].shape)
		# print(spikes.shape)
		AScurrents[i] = (asc_amp[i] + asc_r[i] * AScurrents_temp[i]) * spikes * dt + AScurrents_temp[i] * (1 - asc_k[i] * dt)
	return AScurrents