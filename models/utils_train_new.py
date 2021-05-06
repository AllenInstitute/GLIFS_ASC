from neurons.glif_new import GLIFR
from neurons.synapse_new import Linear
from neurons.network_new import SNNNetwork

import torch
import torch.utils.data as tud

import math
import numpy as np
import matplotlib.pyplot as plt

def sumSinusoid(sim_time, dt, n_sines, periods=None, amplitudes=None, phases=None):
	steps = int(sim_time / dt)

	if periods is None:
		# periods = [np.random.uniform(low=100, high=1000) for i in range(n_sines)]
		periods = [np.random.uniform(low=0.8, high=10) for i in range(n_sines)]
	if amplitudes is None:
		amplitudes = [np.random.uniform(low=0.5, high=4) for i in range(n_sines)]
		# amplitudes = [np.random.uniform(low=0.5, high=2) for i in range(n_sines)]
	if phases is None:
		phases = [np.random.uniform(low=0., high=np.pi * 2) for i in range(n_sines)]
	print(f"using periods: {periods}")
	print(f"using amplitudes: {amplitudes}")
	print(f"using phases: {phases}")

	sines = []
	for i in range(n_sines):
		print(steps)
		print(periods[i] / dt)
		print(steps / (periods[i] / dt))
		x = np.linspace(phases[i], math.pi * 2 * (steps / (periods[i] / dt)) + phases[i], steps)
		sine = np.sin(x) * amplitudes[i] + amplitudes[i] + 4
		sines.append(sine)

	return sum(sines)

def sum_of_sines_target(sim_time, dt, n_sines=4, periods=[1000, 500, 333, 200], weights=None, phases=None, normalize=True):
    '''
    Generate a target signal as a weighted sum of sinusoids with random weights and phases.
    :param n_sines: number of sinusoids to combine
    :param periods: list of sinusoid periods (ms)
    :param weights: weight assigned the sinusoids
    :param phases: phases of the sinusoids
    :return: one dimensional vector of size seq_len contained the weighted sum of sinusoids
    '''
    if periods is None:
        periods = [np.random.uniform(low=100, high=1000) for i in range(n_sines)]
    assert n_sines == len(periods)
    sines = []
    weights = np.random.uniform(low=0.5, high=10, size=n_sines) if weights is None else weights
    phases = np.random.uniform(low=0., high=np.pi * 2, size=n_sines) if phases is None else phases
    for i in range(n_sines):
        sine = np.sin(np.linspace(0 + phases[i], np.pi * 2 * (seq_len // periods[i]) + phases[i], seq_len))
        sines.append(sine * weights[i])

    output = sum(sines)
    if normalize:
        output = output - output[0]
        scale = max(np.abs(np.min(output)), np.abs(np.max(output)))
        output = output / np.maximum(scale, 1e-6)
    return output

def l2_norm(spikes_actual, spikes_target):
	r""" Computes Euclidean distance between actual and target spikes"""
	return torch.dist(spikes_actual, spikes_target, 2)

# def avg_norm(spikes_actual, spikes_target, bnn):
# 	r""" Computes custom loss with Euclidean norm and regularization on km"""
# 	loss = avg_norm(spikes_actual, spikes_target)
# 	print(loss)
# 	# for k in bnn._conns.keys():
# 	# 	loss = loss + 0.01 * torch.mean((1 / bnn._conns[k].weight)** 2)
# 	# loss = loss + 0.01 * torch.mean((1 / bnn.input_linear.weight)** 2)
# 	return loss

def reg(bnn):
	wts = bnn.wts
	loss = 0
	for k in bnn._layers.keys():

		if k[0] != '#':
			# print(k)
			# print(bnn._layers[k].ln_k_m)
			loss = loss + 1 * torch.mean((torch.exp(bnn._layers[k].ln_k_m)) ** 2)
			# loss = loss + 0.01 * torch.mean((torch.exp(bnn._layers[k].thresh)) ** 2)
			
			# for pre, c in bnn._prevs[k]:
			# 	loss = loss + 0.3 * l2_norm(bnn._layers[k].R * torch.exp(bnn._layers[k].ln_k_m) * torch.mean(bnn._conns[c].weight) * (bnn.dt **2) * bnn.sizes[k],torch.ones(bnn.sizes[k]))

	# for k in bnn._conns.keys():
	# 	lyr = bnn._conns[k]
	# 	wt = wts[k]
	# 	# print(f"{k} loss: {torch.mean((100 / (0.5 * wt * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((2 * lyr.weight / wt) ** 2)))}")
	# 	# print(torch.mean(lyr.weight))
	# 	# print(wts)
	# 	loss = loss + torch.mean((100 / (0.5 * wt * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((2 * lyr.weight / wt) ** 2)))
	return loss

def null_reg(bnn, steps, delay, train_spikes_all):
	bnn.reset_state()
	loss = 0
	dt = 0.05
	for k in bnn.conns.keys():
		bnn.reset_state()
		pre, post = bnn.conns[k]
		if post != k:
			linear = bnn._conns[k]

			# 0 to pre
			input_spikes = 10 * torch.ones((steps, 1, 1, bnn.sizes[pre]))
			train_spikes = [torch.zeros(*bnn._layers[post].cells_shape, dtype=torch.float)]
			for i in range(bnn._layers[post].delay - 1):
				train_spikes.append(torch.zeros(*bnn._layers[post].cells_shape, dtype=torch.float))
			for step in range(steps):
				bnn._layers[pre].forward(input_spikes[step,:,:,:])
				y = linear.forward(bnn._layers[pre].spikes_all[-bnn._layers[pre].delay])
				x = bnn._layers[post].forward(y)
				train_spikes.append(x)

			train_spikes = bnn._layers[post].spikes_all
			# print(np.array(train_spikes).shape)
			train_spikes = torch.stack(train_spikes, dim=0)

			# 0 to post
			# 0 to pre
			bnn.reset_state()
			input_spikes_null = torch.zeros((steps, 1, 1, bnn.sizes[post]))
			train_spikes_null = [torch.zeros(*bnn._layers[post].cells_shape, dtype=torch.float)]
			for i in range(bnn._layers[post].delay - 1):
				train_spikes_null.append(torch.zeros(*bnn._layers[post].cells_shape, dtype=torch.float))
			for step in range(steps):
				x = input_spikes_null[step,:,:,:]
				x = bnn._layers[post].forward(x)
				train_spikes_null.append(x)
			# train_spikes_null = bnn._layers[post].spikes_all
			train_spikes_null = torch.stack(train_spikes_null, dim=0)

			# plt.plot(np.arange(len(train_spikes) - delay) * dt, torch.mean(train_spikes_null.detach()[delay:, 0,0,:],-1), label = "average")
			# plt.plot(np.arange(len(train_spikes) - delay) * dt, torch.mean(train_spikes.detach()[delay:, 0,0,:],-1), label = "null")
			
			# plt.legend()
			# plt.show()
			for j in range(steps):
				loss = loss + l2_norm(train_spikes_null[j+delay], train_spikes[j + delay])
	# for k in bnn._layers.keys():
	# 	input_spikes = torch.zeros((steps, 1, 1, bnn.sizes[k]))
	# 	for step in range(steps):
	# 		bnn._layers[k].forward(input_spikes[step,:,:,:])
	# 	train_spikes = bnn._layers[k].spikes_all
	# 	# print(np.array(train_spikes).shape)
	# 	train_spikes = torch.stack(train_spikes, dim=0)

	# 	# plt.plot(np.arange(len(train_spikes) - delay) * dt, torch.mean(train_spikes_all[bnn.layers[-1]].detach()[delay:, 0,0,:],-1), label = "average")
	# 	# plt.plot(np.arange(len(train_spikes) - delay) * dt, torch.mean(train_spikes.detach()[delay:, 0,0,:],-1), label = "null")
		
	# 	# plt.legend()
	# 	# plt.show()
	# 	for j in range(steps):
	# 		loss = loss + l2_norm(train_spikes_all[k][j+delay], train_spikes[j + delay])
	print((loss))
	return 10000000 / loss

def avg_norm(spikes_actual, spikes_target, bnn=None):
	r""" Computes Euclidean distance between actual and target spikes"""
	return torch.dist(torch.mean(spikes_actual,-1).reshape(spikes_target.shape), spikes_target, 2)

def fft_norm(spikes_actual, spikes_target):
	return torch.dist(torch.fft.fft(torch.mean(spikes_actual,-1).reshape(spikes_target.shape)), torch.fft.fft(spikes_target), 2)

def km_reg(bnn, reg_lambda):
	loss = 0
	for k in bnn._layers.keys():
		if k[0] != '#':
			loss = loss + reg_lambda * torch.mean((torch.exp(bnn._layers[k].ln_k_m)) ** 2)
	return loss
def asc_k_reg(bnn):
	loss = 0
	for k in bnn._layers.keys():
		if k[0] != '#':
			loss = loss + 15 * torch.mean((torch.exp(bnn._layers[k].asc_k)) ** 2)
	return loss

def asc_amp_reg(bnn, reg_lambda):
	loss = 0
	for k in bnn._layers.keys():
		if k[0] != '#':
			loss = loss + reg_lambda * torch.mean((bnn._layers[k].asc_amp) ** 2)
	return loss

def add_norm(spikes_actual, spikes_target):
	r""" Computes Euclidean distance between actual and target spikes"""
	# print(spikes_actual)
	# print(spikes_target)
	shape = spikes_actual.shape
	loss = 0.0
	for i in range(shape[-1]):
		loss = loss + torch.dist(spikes_actual[:,:,i].reshape(spikes_target.shape), spikes_target, 2)
	return loss / shape[-1]

def generate_sinusoid(gamma, freq, sim_time, dt, amplitude = 5):
	r""" Generates sinusoid target with max value gamma / 1000
	freq given in kHz (1/ms)
	"""
	inp = torch.arange(start = 0, end = sim_time, step = dt) * 2 * math.pi
	out = torch.sin(inp * (freq)) * 1
	out = out + 10
	return out

def constrain(grad, c = 1e2):
	if torch.linalg.norm(torch.mean(grad)) > c:
		print("clipping")
		grad = 20 * grad / torch.linalg.norm(grad)
	return grad