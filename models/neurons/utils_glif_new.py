# This code defines utilities (state updates) needed for the GLIFR

import torch
import torch.nn.functional as F

def exponential_spiking(voltage, thresh, spike_r, sigma_v, dt, rnn=False):
	return spike_r * torch.sigmoid((voltage - thresh) / sigma_v)

def linear_spiking(voltage, thresh, spike_r, sigma_v, dt):
	return F.relu((voltage - thresh) / sigma_v)

def voltage_update(voltage, v_reset, k_m, R, I0, AScurrents, SYNcurrents, spikes, dt, rnn=False):
	currents_sum = I0 + (SYNcurrents) + torch.sum(AScurrents, dim=0)
	rates = spikes * dt
	if rnn:
		v_delta = -dt * k_m * voltage
	else:
		v_delta = (-dt * k_m - rates) * voltage + rates * v_reset
	v_delta = v_delta + R * k_m * currents_sum * dt
	if rnn:
		voltage = voltage + v_delta
	else:
		voltage = voltage + v_delta
	return voltage

def SYNcurrents_update_current(SYNcurrents, incoming, k_syn, dt, rnn=False):
	# SYNcurrents = SYNcurrents - k_syn * SYNcurrents * dt + (incoming * dt)
	if rnn:
		k_syn = 1 / dt
	return SYNcurrents - k_syn * SYNcurrents * dt + (incoming * dt)

def AScurrents_update(AScurrents, spikes, asc_amp, asc_k, asc_r, dt, rnn=False):
	# print(spikes.shape)
	# print(((asc_amp + asc_r).reshape((2,1,1,1))*AScurrents).shape)
	# print((torch.matmul((asc_amp + asc_r).reshape((2,1,1,1))*AScurrents, spikes)).shape)
	# return ((asc_amp + asc_r).reshape((2,1,1,1))*AScurrents * spikes) * dt + AScurrents * (1 - asc_k.reshape((2,1,1,1)) * dt)
	AScurrents_temp = AScurrents.clone()
	if rnn:
		AScurrents = AScurrents * 0
	else:
		for i in range(len(asc_amp)):
			# print(spikes.shape)
			# print(AScurrents[i].shape)
			# print(spikes.shape)
			AScurrents[i] = (asc_amp[i] + asc_r[i] * AScurrents_temp[i]) * spikes * dt + AScurrents_temp[i] * (1 - asc_k[i] * dt)
	return AScurrents