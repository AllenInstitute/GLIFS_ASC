import numpy as np
import statistics as stat

import torch
import torch.nn as nn
import torch.utils.data as tud


def create_sines(sim_time, dt, amp, noise_mean, noise_std, freqs):
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

    Returns
    -------
    Numpy Array(nsteps, len(freqs))
        input sequences
    Numpy Arryay(nsteps, len(freqs))
        target sequences
    """
    n = len(freqs)
    nsteps = int(sim_time / dt)
    time = np.arange(start = 0, stop = sim_time, step = dt)

    targets = np.empty((nsteps, n))
    inputs = np.empty((nsteps, n))

    for i in range(n):
        offset = (i / n) + 0.25
        noise = np.random.normal(noise_mean, noise_std, nsteps)
        freq = 2 * np.pi * freqs[i]
        
        targets[:, i] = amp * np.sin(freq * time) + noise + offset
        inputs[i] = offset
    
    return inputs, targets

def create_dataset(inputs, targets):
    nsteps, n = inputs.shape
    inputs = inputs.reshape((n, nsteps, 1, 1))
    targets = targets.reshape((n, nsteps, 1, 1))
    return tud.TensorDataset(torch.tensor(targets).float(), torch.tensor(targets).float())

def train_rbnn(model, traindataset, batch_size, num_epochs, lr, reg_lambda, verbose = True, predrive = True):
    """
    Train RBNN model using trainloader and track metrics.

    Parameters
    ----------
    model : RBNN
        network to be trained
    trainloader : Torch Dataset
        training dataset with input, target pairs
    batch_size : int
        size of batches to be used during training
    num_epochs : int
        number of epochs to train the model for
    lr : float
        learning rate to use
    reg_lambda : float
        regularization constant to use for time constant
    verbose : boolean, optional, default True
        whether to print loss
    predrive : boolean, optional, default True
        whether to apply predrive in training
    
    Returns
    -------
    training_info : Dict[Str: Any]
        dictionary including following information
        - losses : List[float]
            training loss over epochs
        - init_outputs : List[List[]]
            list where ith element is a list that is the output of 
            the network in response to the ith input before training
        - final_outputs : List[List[]]
            list where the ith element is a list that is the output of
            the network in response to the ith input before training
        - weights : List[List[List[float]]]
            ith list corresponds to a single linear layer 
                ordered input_linear, rec_linear, output_linear
            jth list of ith list corresponds to jth epoch
            kth element of the jth list corresponds to a weight
        - threshes : List[List[float]]
            ith element is a list of thresholds of neurons
            at the ith epoch
        - k_ms: List[List[float]]
            ith element is a list of k_ms of neurons at the
            ith epoch
        - asc_amps : List[List[float]]
            ith element is a list of asc_amps of neurons at
            the ith epoch
        - asc_rs : List[List[float]]
            ith element is a list of asc_rs of neurons at
            the ith epoch
        - weight_grads : List[List[List[float]]]
            ith element is a list corresponding to the ith epoch
            jth element of ith list corresponds to a single linear layer 
                ordered input_linear, rec_linear, output_linear
            kth element of the jth list corresponds to a weight gradient
        - thresh_grads : List[List[float]]
            ith element is a list of threshold gradients of neurons
            at the ith epoch
        - k_m_grads: List[List[float]]
            ith element is a list of k_m gradients of neurons at the
            ith epoch
        - asc_amp_grads : List[List[float]]
            ith element is a list of asc_amp gradients of neurons at
            the ith epoch
        - asc_r_grads : List[List[float]]
            ith element is a list of asc_r gradients of neurons at
            the ith epoch
    """
    training_info = {"losses": [],
                    "weights": [[],[],[]],
                    "threshes": [],
                    "k_ms": [],
                    "asc_amps": [],
                    "asc_rs": [],
                    "weight_grads": [[],[],[]],
                    "thresh_grads": [],
                    "k_m_grads": [],
                    "asc_amp_grads": [],
                    "asc_r_grads": []}
    model.eval()
    init_dataloader = tud.DataLoader(traindataset, batch_size=1, shuffle=False)
    init_outputs = []
    for batch_ndx, sample in enumerate(init_dataloader):
        inputs, targets = sample
        _, nsteps, _, _ = inputs.shape
        model.reset_state()
        init_outputs.append(model.forward(inputs)[:, -nsteps:, :, :])
    training_info["init_outputs"] = init_outputs

    init_outputs_driven = []
    for batch_ndx, sample in enumerate(init_dataloader):
        inputs, targets = sample
        _, nsteps, _, _ = inputs.shape
        model.reset_state()

        model(targets) # Predrive
        init_outputs_driven.append(model.forward(inputs)[:, -nsteps:, :, :])
    training_info["init_outputs_driven"] = init_outputs_driven

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    trainloader = tud.DataLoader(traindataset, batch_size = batch_size, shuffle = True)
    model.train()
    for epoch in range(num_epochs):
        tot_loss = 0
        tot_pairs = 0
        loss_batch = []
        for batch_ndx, sample in enumerate(trainloader):
            loss = 0.0
            inputs, targets = sample

            # a, b, c, d, e = inputs.shape
            # inputs = inputs.reshape(b, a, d, e)

            # a, b, c, d, e = targets.shape
            # targets = targets.reshape(b, a, d, e)
            _, nsteps, _, _ = inputs.shape
            tot_pairs += len(targets)

            model.reset_state(len(targets))
            optimizer.zero_grad()

            if predrive:
                with torch.no_grad():
                    model(targets)

            # outputs = torch.stack(model(inputs)[-nsteps:], dim=0)
            outputs = model(inputs)
            outputs = outputs[:, -nsteps:, :, :]

            loss = loss + loss_fn(outputs, targets)
            loss = loss + km_reg(model, reg_lambda)
            loss.backward()

            optimizer.step()
            tot_loss += loss.item()
            loss_batch.append(loss.item() / len(targets))
        if verbose:
            print(f"epoch {epoch}/{num_epochs}: loss of {tot_loss / tot_pairs} with variance {0 if len(loss_batch) < 2 else stat.variance(loss_batch)}")

        training_info["losses"].append(tot_loss)
        training_info["weights"][0].append([model.input_linear.weight[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.in_size)])
        training_info["weights"][1].append([model.rec_linear.weight[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.hid_size)])
        training_info["weights"][2].append([model.output_linear.weight[i,j].item() + 0.0 for i in range(model.out_size) for j in range(model.hid_size)])
        training_info["k_ms"].append([torch.exp(model.neuron_layer.ln_k_m[0,0,j])  + 0.0 for j in range(model.hid_size)])
        training_info["threshes"].append([model.neuron_layer.thresh[0,0,j]  + 0.0 for j in range(model.hid_size)])
        training_info["asc_amps"].append([model.neuron_layer.asc_amp[j,0,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
        training_info["asc_rs"].append([model.neuron_layer.asc_r[j,0,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])

        training_info["weight_grads"][0].append([model.input_linear.weight.grad[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.in_size)])
        training_info["weight_grads"][1].append([model.rec_linear.weight.grad[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.hid_size)])
        training_info["weight_grads"][2].append([model.output_linear.weight.grad[i,j].item() + 0.0 for i in range(model.out_size) for j in range(model.hid_size)])
        training_info["k_m_grads"].append([model.neuron_layer.ln_k_m.grad[0,0,j]  + 0.0 for j in range(model.hid_size)])
        training_info["thresh_grads"].append([model.neuron_layer.thresh.grad[0,0,j]  + 0.0 for j in range(model.hid_size)])
        training_info["asc_amp_grads"].append([model.neuron_layer.asc_amp.grad[j,0,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
        training_info["asc_r_grads"].append([model.neuron_layer.asc_r.grad[j,0,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
    
    final_outputs = []
    model.eval()
    for batch_ndx, sample in enumerate(init_dataloader):
        inputs, _ = sample
        _, nsteps, _, _ = inputs.shape
        model.reset_state()
        final_outputs.append(model.forward(inputs)[:, -nsteps:, :, :])
    training_info["final_outputs"] = final_outputs

    final_outputs_driven = []
    for batch_ndx, sample in enumerate(init_dataloader):
        inputs, targets = sample
        _, nsteps, _, _ = inputs.shape
        model.reset_state(batch_size = 1)

        model(targets) # Predrive first dimension is batch so 1
        final_outputs_driven.append(model.forward(inputs[:, -nsteps:, :, :]))
    training_info["final_outputs_driven"] = final_outputs_driven
    return training_info

def km_reg(rbnn, reg_lambda):
	return reg_lambda * torch.mean((torch.exp(rbnn.neuron_layer.ln_k_m)) ** 2)



#     train_output_data = data[:, j]#grab a single sinusoid
#             #test_output_data = data[j]
            
#             # training, validation, and testing data (organized by batch)
#             batched_train_data =  torch.tensor(formatModelData(stim[:, j], input_length, batch_size), device=device)
            
#             #batched_test_data = torch.tensor(formatModelData(stim[j], input_length, batch_size), device=device)
            
#             batched_train_output_data = torch.tensor(formatModelData(train_output_data, input_length, batch_size), device=device)
    
#             #batched_test_output_data = torch.tensor(formatModelData(test_output_data, input_length, batch_size), device=device)
            
#             train_dataset = TensorDataset(batched_train_data, batched_train_output_data) # create your datset
#             train_dataloader = DataLoader(train_dataset, batch_size=None) # create your dataloader


# def formatModelData(data, seq_length, batch_size):
#     """Prepares data for training by reshaping it in batches. (batch_count, mini-batch count, input_length,  input_size = dim).
    
#     :param data: numpy array of time series data with each column corresponding to a data feature
#     :param seq_length: the number of time steps the model sees to train on per mini-batch.
#     :param batch_size: the number of mini-batches of seq_length the model sees to train on per batch.

#     :return data_batched: numpy array of the data shaped as (batch_count, mini-batch count, input_length,  input_size = dim)
#     """
#     r = len(data)#get shape of data, rows = number of time steps, cols = number of features

#     mini_batch_data = []#list to hold each mini-batch of data as it is iteratively collected
#     data_mini_batched = []#list hold the mini-batches
#     for i in range(r):
#         mini_batch_data.append(data[i])#grab data for each step in the mini-batch
#         if (i+1) % seq_length == 0:
#             data_mini_batched.append(np.vstack(mini_batch_data))#append each data input instance
#             mini_batch_data = []#clear list of per mini-batch data
            
    
#     data_mini_batched = np.stack(data_mini_batched, axis=0)#convert list of mini-batches to np.array by adding mini-batch count axis
    
#     batch_data = []
#     data_batched = []
#     for i in range(len(data_mini_batched)):
#         batch_data.append(data_mini_batched[i, :])#grab mini-batched data for each mini-batch in the batch
#         if (i+1) % batch_size == 0:
#             data_batched.append(np.stack(batch_data, axis=0))#append each data input instance
#             batch_data = []#clear list of per batch data
            
    
#     data_batched = np.stack(data_batched, axis=0)#convert list of batches to np.array along the batch axis
        
        
#     return data_batched


#     sin_frequencies = 10 ** np.linspace(np.log10(fmin), np.log10(fmax), num=dim)



# def (t, amp, noise_u, noise_std, freq_vals):
#     """Creates an N-dimension sinusoid data set.
    
#     :param dim: value indicating the dimension of the sinusoid data
#     :param t: numpy array of time points for the sinusoid data to be generated at
#     :param amp: scalar indicating the sinusoidal amplitude
#     :param noise_u: mean noise distribution value
#     :param noise_std: standard deviation for noise distribution
#     :param freq_vals: numpy array of the frequency values for each dimension of sinusoid data
    
#     :return data: numpy array of sinusoidal data with each column representing a dimension
#     """
#     data = np.empty((len(t), len(freq_vals)))#preallocate space for sinusoidal data
#     stim = np.empty((len(t), len(freq_vals)))#preallocate space for the input stimulus
#     for i in range(len(freq_vals)):#for each dimension up to 'dim'
#         offset = i/len(freq_vals) + 0.25
#         noise = np.random.normal(noise_u, noise_std, len(t))#grab a noise value from distribution
#         f = freq_vals[i]#grab a frequency value from the range indicated
#         w=2*np.pi*f#calculate angular frequency
#         data[:, i] = amp*np.sin(w*t) + noise + offset#calculate sinusoidal data for each time point
#         stim[i] = offset #input stimulus to the network for training to yield the corresponding sinusoid

#     return data, stim