import numpy as np
import torch
import torch.utils.data as tud


def create_dataset(inputs, targets):
    """
    Creates a torch dataset given inputs and targets
    
    Parameters
    ----------
    inputs : numpy array (nsteps, n, input_size)
        numpy array containing n input samples, each with
        nsteps timesteps and a dimension of input_size
    targets : numpy array (nsteps, n, target_size)
        numpy array containing n target samples, each with
        nsteps timesteps and a dimension of input_size
        and corresponding to inputs
    """
    nsteps, n, input_size = inputs.shape

    inputs = np.moveaxis(inputs, 0,1)
    targets = np.moveaxis(targets, 0,1)

    inputs = inputs.reshape((n, nsteps, input_size))
    targets = targets.reshape((n, nsteps, -1))

    return tud.TensorDataset(torch.tensor(inputs).float(), torch.tensor(targets).float())

def create_multid_pattern(sim_time, dt, amp, noise_mean, noise_std, freqs, input_size):
    """
    Create a dataset of three-dimensional patterns based on Task 1.1 Pattern Generation in https://arxiv.org/pdf/1901.09049.pdf
    Parameters
    ----------
    sim_time : float
        number of ms of total simulation time
    dt : float
        number of ms in each timestep
    amp : float
        amplitude of the target
    noise_mean : float
        mean of noise added to target
    noise_std : float
        std of noise added to target
    freqs : List
        list of frequencies (1/ms or kHz) of sinusoids
    Returns
    -------
    Numpy Array(nsteps, len(freqs))
        input sequences
    Numpy Arryay(nsteps, len(freqs))
        target sequences
    """

    # TODO: should we put in the wave as input? Should we apply an offset to the targets?
    n = len(freqs)
    nsteps = int(sim_time / dt)
    time = np.arange(start = 0, stop = sim_time, step = dt)

    targets = np.empty((nsteps, n))
    inputs = np.ones((nsteps, input_size))

    freq_in = 0.01 # 1/100ms = 0.01/ms

    # for i in range(input_size):
    #     wave = amp * np.sin(2 * np.pi * freq_in * time + (2 * np.pi * i/ input_size))
    #     inputs[:, i] = np.maximum(0,wave)

    for i in range(n):
        offset = (i / n) + 0.25
        noise = np.random.normal(noise_mean, noise_std, nsteps)
        freq = 2 * np.pi * freqs[i]
        
        targets[:, i] = amp * np.sin(freq * time) + noise + offset
        inputs[:, i] = offset

    # Change shape so that inputs and targets have nfreq as output dimension rather than num_samples
    # nsteps, 1, n_out
    inputs = np.expand_dims(inputs, 1)
    targets = np.expand_dims(targets, 1)
    
    return inputs, targets

def create_sines_cued(sim_time, dt, amp, noise_mean, noise_std, freqs, input_size):
    """
    Create a dataset of different frequency sinusoids based on (David Sussillo, Omri Barak, 2013) pattern generation task
    Parameters
    ----------
    sim_time : float
        number of ms of total simulation time
    dt : float
        number of ms in each timestep
    amp : float
        amplitude of the sinusoid
    noise_mean : float
        mean of noise added to sinusoid
    noise_std : float
        std of noise added to sinusoid
    freqs : List
        list of frequencies (1/ms or kHz) of sinusoids
    input_size : int
        number of neurons receiving inputs
    Returns
    -------
    Numpy Array(nsteps, len(freqs), input_size)
        input sequences
    Numpy Arryay(nsteps, len(freqs))
        target sequences
    """
    num_per = int(input_size / len(freqs))

    n = len(freqs)
    nsteps = int(sim_time / dt)
    time = np.arange(start = 0, stop = sim_time, step = dt)

    targets = np.empty((nsteps, n))
    inputs = np.zeros((nsteps, n, input_size))

    for i in range(n):
        offset = (i / n) + 0.25
        noise = np.random.normal(noise_mean, noise_std, nsteps)
        freq = 2 * np.pi * freqs[i]
        
        targets[:, i] = amp * np.sin(freq * time) + noise + offset
        inputs[:,i, num_per * i:num_per * (i + 1)] = amp#offset

    return inputs, targets