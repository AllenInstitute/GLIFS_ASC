import matplotlib
from torch.utils.data.dataset import TensorDataset
# matplotlib.use('Agg')

from hessianfree import HessianFree
import numpy as np
from numpy import random as rnd
import statistics as stat
import matplotlib.pyplot as plt
import random
import sys

import torch
import torch.nn as nn
import torch.utils.data as tud
from torchvision import datasets, transforms
from myterial import salmon, light_green_dark, indigo_light
from pyrnn._plot import clean_axes
from rich.progress import track

def mnist_generator(root, batch_size):
    # def set_header_for(url, filename):
    #     import urllib
    #     opener = urllib.request.URLopener()
    #     opener.addheader('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36')
    #     opener.retrieve(
    #     url, f'{root}/{filename}')

    # set_header_for('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'train-images-idx3-ubyte.gz')
    # set_header_for('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'train-labels-idx1-ubyte.gz')
    # set_header_for('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    # set_header_for('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
    
    # Credit: github.com/locuslab/TCN
    from torchvision import datasets
    new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    datasets.MNIST.resources = [
    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
    for url, md5 in datasets.MNIST.resources
    ]
    train_set = datasets.MNIST(root=root, train=True, download=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
    test_set = datasets.MNIST(root=root, train=False, download=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader

def create_copy(sim_time, dt, noise_mean, noise_std, n):
    """
    Create a dataset of different frequency sinusoids based on (David Sussillo, Omri Barak, 2013) pattern generation task

    Parameters
    ----------
    sim_time : float
        number of ms of total simulation time
    dt : float
        number of ms in each timestep
    noise_mean : float
        mean of noise added to sinusoid
    noise_std : float
        std of noise added to sinusoid
    n : int
        number of input/target sequences

    Returns
    -------
    Numpy Array(nsteps, len(freqs))
        input sequences
    Numpy Arryay(nsteps, len(freqs))
        target sequences
    """
    nsteps = int(sim_time / dt)
    time = np.arange(start = 0, stop = sim_time, step = dt)

    inputs = np.random.randint(1, 8, size = (nsteps, n))
    targets = np.zeros((nsteps, n))

    for i in range(n):
        noise = np.random.normal(noise_mean, noise_std, nsteps)
        
        inputs[10:,i] = 0
        inputs[-11:, i] = 9
        targets[-10:, i] = inputs[0:10, i]
        targets = targets + noise.reshape(noise.shape[0], 1)
    targets = np.expand_dims(targets, -1)
    inputs = np.expand_dims(inputs, -1)
    return inputs, targets

def memory_capacity(sim_time, dt, lookback, noise_mean=0, noise_std=1, input_size=1):
    from scipy.signal import butter,filtfilt

    nsteps = int(sim_time / dt)
    nback = int(lookback / dt)
    targets = noise_mean + noise_std * np.random.randn(nsteps, 1, input_size)
    inputs = noise_mean + noise_std * np.random.randn(nsteps, 1, input_size)

    fs = 1000 / dt
    cutoff = 20
    order = 2
    normal_cutoff = cutoff / (0.5 * fs)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False, fs=fs)
    inputs[:,0,0] = filtfilt(b, a, inputs[:,0,0])

    targets[nback:, :, :] = inputs[:-nback, :, :]
    return inputs, targets

def create_sines(sim_time, dt, amp, noise_mean, noise_std, freqs, input_size):
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

    nsteps_pre = max([min(200,int((1 / (freq)) / dt)) for freq in freqs])
    targets_pre = np.empty((nsteps_pre + 1, n))
    inputs_pre = np.empty((nsteps_pre + 1, n))

    for i in range(n):
        offset = (i / n) + 0.25
        noise = np.random.normal(noise_mean, noise_std, nsteps)
        freq = 2 * np.pi * freqs[i]
        
        targets[:, i] = amp * np.sin(freq * time) + noise + offset
        inputs[:,i] = offset

        
        period = min(int((1 / freqs[i]) / dt), 200)
        noise = np.random.normal(noise_mean, noise_std, period)
        time_pre =np.linspace(-1 / freqs[i], 0, num = period, endpoint = False)# np.arange(start = -1 / freqs[i], stop = 0, step = dt)
        inputs_pre[-period:, i] = offset
        # print((amp * np.sin(freq * time_pre) + noise + offset).shape)
        targets_pre[-period:, i] = (amp * np.sin(2 * np.pi * freqs[i] * time_pre) + noise + offset)[-period:]

    inputs = np.expand_dims(inputs, -1)
    targets = np.expand_dims(targets, -1)
    inputs_pre = np.expand_dims(inputs_pre, -1)
    targets_pre = np.expand_dims(targets_pre, -1)

    return inputs, targets

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
    inputs = np.empty((nsteps, input_size))

    # nsteps_pre = max([min(200,int((1 / (freq)) / dt)) for freq in freqs])
    # targets_pre = np.empty((nsteps_pre + 1, n))
    # inputs_pre = np.empty((nsteps_pre + 1, n))

    freq_in = 0.01 # 1/100ms = 0.01/ms

    for i in range(input_size):
        wave = amp * np.sin(2 * np.pi * freq_in * time + (2 * np.pi * i/ input_size))
        inputs[:, i] = np.maximum(0,wave)

    for i in range(n):
        offset = 0#(i / n) + 0.25
        noise = np.random.normal(noise_mean, noise_std, nsteps)
        freq = 2 * np.pi * freqs[i]
        
        targets[:, i] = amp * np.sin(freq * time) + noise + offset
        # inputs[:,i] = offset

        
        # period = min(int((1 / freqs[i]) / dt), 200)
        # noise = np.random.normal(noise_mean, noise_std, period)
        # time_pre =np.linspace(-1 / freqs[i], 0, num = period, endpoint = False)# np.arange(start = -1 / freqs[i], stop = 0, step = dt)
        # inputs_pre[-period:, i] = offset
        # # print((amp * np.sin(freq * time_pre) + noise + offset).shape)
        # targets_pre[-period:, i] = (amp * np.sin(2 * np.pi * freqs[i] * time_pre) + noise + offset)[-period:]

    # Change shape so that inputs and targets have nfreq as output dimension rather than num_samples
    # nsteps, 1, n_out
    inputs = np.expand_dims(inputs, 1)
    targets = np.expand_dims(targets, 1)
    # inputs_pre = np.expand_dims(inputs_pre, -1)
    # targets_pre = np.expand_dims(targets_pre, -1)

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

def create_dataset(inputs, targets, input_size = 1, target_size = 1):
    nsteps, n, _ = inputs.shape
    # nsteps_pre, n_pre, _ = inputs_pre.shape

    inputs = np.moveaxis(inputs, 0,1)
    targets = np.moveaxis(targets, 0,1)
    # inputs_pre = np.moveaxis(inputs_pre, 0,1)
    # targets_pre = np.moveaxis(targets_pre, 0,1)

    inputs = inputs.reshape((n, nsteps, input_size))
    targets = targets.reshape((n, nsteps, target_size))

    # print(inputs_pre.shape)
    # inputs_pre = inputs_pre.reshape((n_pre, nsteps_pre, input_size))
    # targets_pre = targets_pre.reshape((n_pre, nsteps_pre, target_size))
    return tud.TensorDataset(torch.tensor(inputs).float(), torch.tensor(targets).float())#, torch.tensor(inputs_pre).float(), torch.tensor(targets_pre).float())

def train_rbnn_mnist(model, batch_size, num_epochs, lr, glifr, verbose = True):#, batch_size, num_epochs, lr, reg_lambda, verbose = True, predrive = True, glifr = True, task = "pattern"):
    """
    Train RBNN model using trainloader and track metrics.

    Parameters
    ----------
    model : RBNN
        network to be trained
    batch_size : int
        size of batches to be used during training
    num_epochs : int
        number of epochs to train the model for
    lr : float
        learning rate to use
    glifr : boolean, optional, default True
        whether to expect fields existing in GLIFR class
    
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
                    "weights": [],
                    "threshes": [],
                    "k_ms": [],
                    "asc_amps": [],
                    "asc_rs": [],
                    "asc_ks": [],
                    "weight_grads": [],
                    "thresh_grads": [],
                    "k_m_grads": [],
                    "asc_amp_grads": [],
                    "asc_r_grads": [],
                    "asc_k_grads": []}
    model.eval()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=1)
    loss_fn = nn.CrossEntropyLoss()
    root = './data/mnist'
    trainloader, testloader = mnist_generator(root, batch_size)
    model.train()
    input_channels = 1
    seq_length = int(784/input_channels)
    # for batch_ndx, sample in enumerate(trainloader):
    for epoch in range(num_epochs):
        # if epoch % 1 == 0:
        #     trainloader = tud.DataLoader(traindataset, batch_size = batch_size, shuffle = True)
        tot_loss = 0
        tot_pairs = 0
        loss_batch = []
        reg_lambda = 0.01
        for batch_ndx, (data,target) in enumerate(trainloader):
            target = target.long()

            # print(batch_ndx)
            n_subiter = 1
            if batch_ndx % 100 == 0 and batch_ndx > 0:
                print(f"loss of {loss_batch[-1]} on batch {batch_ndx}/{len(trainloader)}")
            for i in range(n_subiter):
                loss = 0.0
                # print(data.shape)
                # data = data.view(-1, 28, 28)
                data = data.view(-1, 28 * 28, 1)

                optimizer.zero_grad()

                _, nsteps, _ = data.shape
                # with torch.no_grad():
                #     for i in range(10):
                #         a = random.randint(0, nsteps - 5)
                #         inputs[:, a:a+5, :] = 0
                tot_pairs += len(target)

                if True:#batch_ndx == 0:
                    model.reset_state(len(target))
                optimizer.zero_grad()

                outputs = model(data)
                outputs = outputs.reshape(len(target), 10, 28 * 28)[:,:,-1]
                # outputs = outputs.reshape(len(target), 10, 28)[:,:,-1]#torch.mean(outputs.reshape(len(target), 10, 28), -1)
                loss = loss + loss_fn(outputs, target)
                # if i % n_subiter == 0:
                #     print(loss.item() / len(targets))
                if glifr:
                    loss = loss + aa_reg(model, reg_lambda = reg_lambda)
                    reg_lambda *= 0.9
                # if glifr:
                #     loss = loss + km_reg(model, reg_lambda)
                loss.backward()
                if glifr:
                    pass
                    # # print(f"weight: {torch.mean(model.neuron_layer.weight_iv.grad)}")
                    # # print(f"v_reset: {torch.mean(model.neuron_layer.v_reset.grad)}")
                    # print(f"lnkm: {torch.mean(model.neuron_layer.ln_k_m.grad)}")
                    # print(f"lnasck: {torch.mean(model.neuron_layer.ln_asc_k.grad)}")
                    # print(f"thresh: {torch.mean(model.neuron_layer.thresh.grad)}")
                    # print(f"ascr: {torch.mean(model.neuron_layer.asc_r.grad)}")
                    # print(f"ascamp: {torch.mean(model.neuron_layer.asc_amp.grad)}")
                    # print(f"weight_out: {torch.mean(model.output_linear.weight.grad)}")
                # else:
                #     print(f"weight_ih: {torch.mean(model.neuron_layer.weight_ih.grad)}")
                #     print(f"weight_hh: {torch.mean(model.neuron_layer.weight_hh.grad)}")
                #     print(f"weight_out: {torch.mean(model.output_linear.weight.grad)}")

                
                if glifr:
                    with torch.no_grad():
                        # model.neuron_layer.thresh.grad *= 0
                        # model.neuron_layer.ln_k_m.grad *= 0
                        # model.neuron_layer.asc_amp.grad *= 0
                        # model.neuron_layer.asc_r.grad *= 0
                        # model.neuron_layer.ln_asc_k.grad *= 0
                        # model.neuron_layer.weight_iv.grad *= 0
                        # model.output_linear.weight.grad *= 0
                        pass

                optimizer.step()
                # if epoch % 2 == 0 and epoch < 20 and i % n_subiter == 0:
                # scheduler.step()
                
                tot_loss += loss.item()
                loss_batch.append(loss.item() / len(target))
        if verbose:
            print(f"epoch {epoch}/{num_epochs}: loss of {tot_loss / tot_pairs} with variance {0 if len(loss_batch) < 2 else stat.variance(loss_batch)}")

        training_info["losses"].append(tot_loss)
        # training_info["weights"][0].append([model.input_linear.weight[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.in_size)])
        # if glifr:
        #     training_info["weights"][1].append([model.rec_linear.weight[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.hid_size)])
        # training_info["weights"][2].append([model.output_linear.weight[i,j].item() + 0.0 for i in range(model.out_size) for j in range(model.hid_size)])

        if glifr:
            training_info["k_ms"].append([torch.exp(model.neuron_layer.ln_k_m[0,j])  + 0.0 for j in range(model.hid_size)])
            training_info["threshes"].append([model.neuron_layer.thresh[0,j]  + 0.0 for j in range(model.hid_size)])
            training_info["asc_ks"].append([torch.exp(model.neuron_layer.ln_asc_k[j,0,m])  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_amps"].append([model.neuron_layer.asc_amp[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_rs"].append([model.neuron_layer.asc_r[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            # training_info["weights"].append([torch.mean(model.neuron_layer.weight_iv[:,m])  + 0.0 for m in range(model.hid_size)])

        # TODO: Replace with real weight
        # training_info["weight_grads"][0].append([model.input_linear.weight.grad[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.in_size)])
        # if glifr:
        #     training_info["weight_grads"][1].append([model.rec_linear.weight.grad[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.hid_size)])
        # training_info["weight_grads"][2].append([model.output_linear.weight.grad[i,j].item() + 0.0 for i in range(model.out_size) for j in range(model.hid_size)])
        if glifr and epoch % 10 == 0:
            training_info["k_m_grads"].append([model.neuron_layer.ln_k_m.grad[0,j]  + 0.0 for j in range(model.hid_size)])
            training_info["thresh_grads"].append([model.neuron_layer.thresh.grad[0,j]  + 0.0 for j in range(model.hid_size)])
            training_info["asc_k_grads"].append([model.neuron_layer.ln_asc_k.grad[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_amp_grads"].append([model.neuron_layer.asc_amp.grad[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_r_grads"].append([model.neuron_layer.asc_r.grad[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            # training_info["weight_grads"].append([torch.mean(model.neuron_layer.weight_iv.grad[:,m])  + 0.0 for m in range(model.hid_size)])

    final_outputs = []
    model.eval()
    # torch.save(model.state_dict(), "trained_model.pt")
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            target = target.long()
            model.reset_state(len(target))
            # target = torch.unsqueeze(target, -1)
            # data = data.view(-1, 28, 28)
            data = data.view(-1, 28 * 28, 1)
            output = model(data)
            output = output.reshape(len(target), 10, 28 * 28)[:,:,-1]
            # output = output.reshape(len(target), 10, 28)[:,:,-1]
            # output = torch.mean(output.reshape(len(target), 10, 28), -1)
            test_loss += loss_fn(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            # pred = torch.argmax(torch.sigmoid(output), 1, keepdim=True).long()
            print(pred.shape)
            print(target.shape)
            print(data.shape)
            print(correct)
            
            # pred = torch.sigmoid(output).data.max(1, keepdim=True)[1]
            correct += (pred == ((target.data.view_as(pred)))).sum()
    test_loss = test_loss * 1.0 / len(testloader.dataset)
    
    print(f"loss: {test_loss}")
    print(f"accuracy: {correct * 1.0 / len(testloader.dataset)}")

    original_stdout = sys.stdout # Save a reference to the original standard output

    with open('results.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(f"loss: {test_loss}")
        print(f"accuracy: {correct * 1.0 / len(testloader.dataset)}")
        sys.stdout = original_stdout

    return training_info

def train_rbnn_copy(model, batch_size, num_epochs, lr, glifr, nrepeat, output_size, verbose = True):#, batch_size, num_epochs, lr, reg_lambda, verbose = True, predrive = True, glifr = True, task = "pattern"):
    """
    Train RBNN model using trainloader and track metrics.

    Parameters
    ----------
    model : RBNN
        network to be trained
    batch_size : int
        size of batches to be used during training
    num_epochs : int
        number of epochs to train the model for
    lr : float
        learning rate to use
    glifr : boolean, optional, default True
        whether to expect fields existing in GLIFR class
    
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
                    "weights": [],
                    "threshes": [],
                    "k_ms": [],
                    "asc_amps": [],
                    "asc_rs": [],
                    "asc_ks": [],
                    "weight_grads": [],
                    "thresh_grads": [],
                    "k_m_grads": [],
                    "asc_amp_grads": [],
                    "asc_r_grads": [],
                    "asc_k_grads": []}
    model.eval()
    # model = model.float()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=1)
    loss_fn = nn.MSELoss()
    
    model.train()
    input_channels = 1
    # for batch_ndx, sample in enumerate(trainloader):
    num_epochs = 0
    sl_last = 0
    ntry = 128
    num_epochs_each = 200

    trainloaders = []
    testloaders = []

    for sl in range(50):
        inputs = np.zeros((ntry, nrepeat * (sl + 1), output_size))
        targets = np.zeros((ntry, nrepeat * (sl + 1), output_size))

        if sl > 0:
            inputs[:, :sl, :] = np.random.randint(0, 2, inputs[:, :sl, :].shape)
            # print(np.random.choice(inputs.shape[0]))
            # inputs[np.random.choice(inputs.shape[0]), np.random.choice(sl), np.random.choice(inputs.shape[-1])] = 1
            # inputs[:, sl, :] = 0
        
        for rep in range(nrepeat - 1):
            targets[:, (rep + 1) * (sl+1):(rep + 2) * (sl+1), :] = inputs[:, 0:sl+1, :]
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()

        trainloaders.append(tud.DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle = True))

        inputs = np.zeros((ntry, nrepeat * (sl + 1), output_size))
        targets = np.zeros((ntry, nrepeat * (sl + 1), output_size))

        if sl > 0:
            inputs[:, :sl, :] = np.random.randint(0, 2, inputs[:, :sl, :].shape)
            inputs[:, sl, :] = 0

        for rep in range(nrepeat - 1):
            targets[:, (rep + 1) * (sl+1):(rep + 2) * (sl+1), :] = inputs[:, 0:sl+1, :]
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()
        testloaders.append(tud.DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle = False))

    def test(seqlen, nrepeat): # TODO: no variation in repeat
        for sl in range(seqlen):
            test_loss = 0
            correct = 0
            total = 0
            corrects = np.zeros(nrepeat - 1)
            totals = np.zeros(nrepeat - 1)
            for batch_ndx, (data, target) in enumerate(testloaders[sl]):
                # print(data.shape)
                # print(target.shape)
                target = target.long()
                model.reset_state(len(target))

                output = model(data)
                # output = output.reshape(len(target) * output_size * nrepeat * (sl + 1), 1)
                # target = target.reshape(len(target) * output_size * nrepeat * (sl + 1))

                test_loss += loss_fn(output, target).item()

                pred = torch.round((output))#.data.max(1, keepdim=True)[1]
                # print(pred.shape)
                # print(target.shape)
                correct += (pred == ((target.data.view_as(pred)))).sum()
                total += len(target) * output_size * nrepeat * (sl + 1)

                for rep in range(nrepeat - 1):
                    corrects[rep] += (target.data[:, (rep + 1) * (sl+1):(rep + 2) * (sl+1), :] == pred.data[:, (rep + 1) * (sl+1):(rep + 2) * (sl+1), :]).sum()
                    totals[rep] += (target.data[:, (rep + 1) * (sl+1):(rep + 2) * (sl+1), :] == target.data[:, (rep + 1) * (sl+1):(rep + 2) * (sl+1), :]).sum()

                # for step in range(nrepeat * (sl + 1)):
                #     corrects[step] += (pred[:, step, :] == ((target[:, step, :].data.view_as(pred[:, step, :])))).sum()
                #     totals[step] += len(target) * output_size
                # pred = torch.argmax(torch.sigmoid(output), 1, keepdim=True).long()
                
            test_loss = test_loss * 1.0 / len(testloaders[sl].dataset)
    
            print(f"loss: {test_loss}")
            print(f"accuracy for {sl} sequence length: {correct * 1.0 / total}...{[corrects[i] / totals[i] for i in range(len(corrects))]}")
                        
    while sl_last < 12 and num_epochs < 2100:
        test(sl_last, nrepeat)
        num_epochs += num_epochs_each
        sl_last += 2
        print(f"on sl {sl_last}")
        
        for epoch in range(num_epochs_each):           
            # if epoch % 1 == 0:
            #     trainloader = tud.DataLoader(traindataset, batch_size = batch_size, shuffle = True)
            tot_loss = 0
            tot_pairs = 0
            loss_batch = []
            reg_lambda = 0.01
            for batch_ndx, (data,target) in enumerate(trainloaders[sl_last]):
                target = target.float()
                data = data.float()
                # print(batch_ndx)
                n_subiter = 1
                if batch_ndx % 100 == 0 and batch_ndx > 0:
                    print(f"loss of {loss_batch[-1]} on batch {batch_ndx}/{len(trainloaders[sl_last])}")
                for i in range(n_subiter):
                    loss = 0.0
                    optimizer.zero_grad()

                    _, nsteps, _ = data.shape
                    tot_pairs += len(target)

                    model.reset_state(len(target))
                    optimizer.zero_grad()

                    outputs = model(data)
                    # print(outputs[0,:,:])
                    # print(target[0,:,:])
                    # print("")
                    # outputs = outputs[:,-1,:]#torch.mean(outputs.reshape(len(target), output_size, (sl_last+1) * nrepeat), -1)
                    loss = loss + loss_fn(outputs, target)
                    # if glifr:
                    #     loss = loss + aa_reg(model, reg_lambda = reg_lambda)
                    #     reg_lambda *= 0.9
                    loss.backward()
                    
                    optimizer.step()
                    tot_loss += loss.item()
                    loss_batch.append(loss.item() / len(target))
            if verbose:
                print(f"epoch {epoch}/{num_epochs_each}: loss of {tot_loss / tot_pairs} with variance {0 if len(loss_batch) < 2 else stat.variance(loss_batch)}")

            training_info["losses"].append(tot_loss)
            # training_info["weights"][0].append([model.input_linear.weight[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.in_size)])
            # if glifr:
            #     training_info["weights"][1].append([model.rec_linear.weight[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.hid_size)])
            # training_info["weights"][2].append([model.output_linear.weight[i,j].item() + 0.0 for i in range(model.out_size) for j in range(model.hid_size)])

            if glifr:
                training_info["k_ms"].append([torch.exp(model.neuron_layer.ln_k_m[0,j])  + 0.0 for j in range(model.hid_size)])
                training_info["threshes"].append([model.neuron_layer.thresh[0,j]  + 0.0 for j in range(model.hid_size)])
                training_info["asc_ks"].append([torch.exp(model.neuron_layer.ln_asc_k[j,0,m])  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
                training_info["asc_amps"].append([model.neuron_layer.asc_amp[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
                training_info["asc_rs"].append([model.neuron_layer.asc_r[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
                # training_info["weights"].append([torch.mean(model.neuron_layer.weight_iv[:,m])  + 0.0 for m in range(model.hid_size)])

            # TODO: Replace with real weight
            # training_info["weight_grads"][0].append([model.input_linear.weight.grad[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.in_size)])
            # if glifr:
            #     training_info["weight_grads"][1].append([model.rec_linear.weight.grad[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.hid_size)])
            # training_info["weight_grads"][2].append([model.output_linear.weight.grad[i,j].item() + 0.0 for i in range(model.out_size) for j in range(model.hid_size)])
            if glifr and epoch % 10 == 0:
                training_info["k_m_grads"].append([model.neuron_layer.ln_k_m.grad[0,j]  + 0.0 for j in range(model.hid_size)])
                training_info["thresh_grads"].append([model.neuron_layer.thresh.grad[0,j]  + 0.0 for j in range(model.hid_size)])
                training_info["asc_k_grads"].append([model.neuron_layer.ln_asc_k.grad[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
                training_info["asc_amp_grads"].append([model.neuron_layer.asc_amp.grad[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
                training_info["asc_r_grads"].append([model.neuron_layer.asc_r.grad[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
                # training_info["weight_grads"].append([torch.mean(model.neuron_layer.weight_iv.grad[:,m])  + 0.0 for m in range(model.hid_size)])

    final_outputs = []
    model.eval()
    # torch.save(model.state_dict(), "trained_model.pt")
    
    test(sl_last, nrepeat)
    return training_info

def train_rbnn(model, traindataset, batch_size, num_epochs, lr, reg_lambda, verbose = True, predrive = True, glifr = True, task = "pattern", decay=False):
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
    glifr : boolean, optional, default True
        whether to expect fields existing in GLIFR class
    
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
    if task == "copy":
        with torch.no_grad():
            data_gen = traindataset()
            traindataset = next(data_gen)

    training_info = {"losses": [],
                    "weights": [],
                    "threshes": [],
                    "k_ms": [],
                    "asc_amps": [],
                    "asc_rs": [],
                    "asc_ks": [],
                    "weight_grads": [],
                    "thresh_grads": [],
                    "k_m_grads": [],
                    "asc_amp_grads": [],
                    "asc_r_grads": [],
                    "asc_k_grads": []}
    model.eval()
    if task == "pattern":
        init_dataloader = tud.DataLoader(traindataset, batch_size=1, shuffle=False)
        init_outputs = []
        for batch_ndx, sample in enumerate(init_dataloader):
            inputs, targets = sample
            _, nsteps, _ = inputs.shape
            model.reset_state()
            init_outputs.append(model.forward(inputs)[:, -nsteps:, :])
        training_info["init_outputs"] = init_outputs

        init_outputs_driven = []
        for batch_ndx, sample in enumerate(init_dataloader):
            inputs, targets = sample
            _, nsteps, _ = inputs.shape
            model.reset_state()

            model(targets) # Predrive
            # model(targets_pre) # Predrive
            init_outputs_driven.append(model.forward(inputs)[:, -nsteps:, :])
        training_info["init_outputs_driven"] = init_outputs_driven
    elif task == "pattern_multid":
        init_dataloader = tud.DataLoader(traindataset, batch_size=1, shuffle=False)
        init_outputs = []
        for batch_ndx, sample in enumerate(init_dataloader):
            inputs, targets = sample
            _, nsteps, _ = inputs.shape
            model.reset_state()
            outputs = model.forward(inputs)
            for dim in range(outputs.shape[-1]):
                init_outputs.append(outputs[:,:,dim])
        training_info["init_outputs"] = init_outputs

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.SmoothL1Loss()
    trainloader = tud.DataLoader(traindataset, batch_size = batch_size, shuffle = True)
    model.train()
    # for batch_ndx, sample in enumerate(trainloader):
    for epoch in range(num_epochs):
        # if epoch % 1 == 0:
        #     trainloader = tud.DataLoader(traindataset, batch_size = batch_size, shuffle = True)
        if task == "copy":
            with torch.no_grad():
                traindataset = next(data_gen)
                trainloader = tud.DataLoader(traindataset, batch_size, shuffle = True)
        tot_loss = 0
        tot_pairs = 0
        loss_batch = []
        # if epoch % 10 == 0:
        #     print(model.neuron_layer.ln_asc_k)
        #     print(model.neuron_layer.ln_asc_k.grad) # -4 and -1 isn't good..-1 and 0 isn't good
        # for epoch in range(num_epochs):
        reg_lambda = 0.01
        for batch_ndx, sample in enumerate(trainloader):
            # print(batch_ndx)
            n_subiter = 1
            for i in range(n_subiter):
                loss = 0.0
                inputs, targets = sample
                
                # plt.plot(inputs[0,:,0].detach().numpy())
                # plt.plot(targets[0,:,0].detach().numpy())
                # plt.show()
                # inputs = inputs.detach()
                # targets = targets.detach()

                _, nsteps, _ = inputs.shape
                # with torch.no_grad():
                #     for i in range(10):
                #         a = random.randint(0, nsteps - 5)
                #         inputs[:, a:a+5, :] = 0
                tot_pairs += len(targets)

                if True:#batch_ndx == 0:
                    model.reset_state(len(targets))
                optimizer.zero_grad()

                # if predrive:
                #     with torch.no_grad():
                #         model(targets)
                # if False:#predrive:
                #     with torch.no_grad():
                #         model(targets_pre)
                        # plt.plot(targets_pre[0,:,0])
                        # plt.show()

                # outputs = torch.stack(model(inputs)[-nsteps:], dim=0)
                outputs = model(inputs)
                outputs = outputs[:, -nsteps:, :]
                #print(outputs.shape)
                # plt.plot(outputs[0,:,0].detach().numpy())
                # plt.plot(targets[0,:,0].detach().numpy())
                # plt.show()

                if task == "copy" and epoch == num_epochs - 1:
                    np.savetxt('outputs_rnn_1000lng.txt', torch.squeeze(outputs).detach().numpy(), fmt="%i")
                    np.savetxt('targets_rnn_1000lng.txt', torch.squeeze(targets).detach().numpy(), fmt="%i")
                    np.savetxt('inputs_rnn_1000lng.txt', torch.squeeze(inputs).detach().numpy(), fmt="%i")

                loss = loss + loss_fn(outputs, targets)
                null_loss = loss_fn(outputs*0, targets)
                # if i % n_subiter == 0:
                #     print(loss.item() / len(targets))
                if glifr:
                    #loss = loss + aa_reg(model, reg_lambda = reg_lambda)
                    reg_lambda *= 0.9
                # if glifr:
                #     loss = loss + km_reg(model, reg_lambda)
                loss.backward()
                
                if glifr:
                    with torch.no_grad():
                        # model.neuron_layer.thresh.grad *= 0
                        # model.neuron_layer.ln_k_m.grad *= 0
                        # model.neuron_layer.asc_amp.grad *= 0
                        # model.neuron_layer.asc_r.grad *= 0
                        # model.neuron_layer.ln_asc_k.grad *= 0
                        # model.neuron_layer.weight_iv.grad *= 0
                        # model.output_linear.weight.grad *= 0
                        pass

                optimizer.step()
                # if epoch % 2 == 0 and epoch < 20 and i % n_subiter == 0:
                if decay:# and epoch < 150:
                    scheduler.step()
                
                tot_loss += loss.item()
                loss_batch.append(loss.item() / len(targets))
        if verbose:
            print(f"epoch {epoch}/{num_epochs}: loss of {tot_loss / tot_pairs} with variance {0 if len(loss_batch) < 2 else stat.variance(loss_batch)} and null loss {null_loss}")

        training_info["losses"].append(tot_loss)
        # training_info["weights"][0].append([model.input_linear.weight[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.in_size)])
        # if glifr:
        #     training_info["weights"][1].append([model.rec_linear.weight[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.hid_size)])
        # training_info["weights"][2].append([model.output_linear.weight[i,j].item() + 0.0 for i in range(model.out_size) for j in range(model.hid_size)])

        if glifr:
            training_info["k_ms"].append([torch.exp(model.neuron_layer.ln_k_m[0,j])  + 0.0 for j in range(model.hid_size)])
            training_info["threshes"].append([model.neuron_layer.thresh[0,j]  + 0.0 for j in range(model.hid_size)])
            training_info["asc_ks"].append([torch.exp(model.neuron_layer.ln_asc_k[j,0,m])  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_amps"].append([model.neuron_layer.asc_amp[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_rs"].append([model.neuron_layer.asc_r[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            # training_info["weights"].append([torch.mean(model.neuron_layer.weight_iv[:,m])  + 0.0 for m in range(model.hid_size)])

        # TODO: Replace with real weight
        # training_info["weight_grads"][0].append([model.input_linear.weight.grad[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.in_size)])
        # if glifr:
        #     training_info["weight_grads"][1].append([model.rec_linear.weight.grad[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.hid_size)])
        # training_info["weight_grads"][2].append([model.output_linear.weight.grad[i,j].item() + 0.0 for i in range(model.out_size) for j in range(model.hid_size)])
        if glifr and epoch % 10 == 0:
            print(torch.mean(model.neuron_layer.ln_k_m.grad))
            print(torch.mean(model.neuron_layer.thresh.grad))
            print(torch.mean(model.neuron_layer.ln_asc_k.grad))
            print(torch.mean(model.neuron_layer.asc_amp.grad))
            training_info["k_m_grads"].append([model.neuron_layer.ln_k_m.grad[0,j]  + 0.0 for j in range(model.hid_size)])
            training_info["thresh_grads"].append([model.neuron_layer.thresh.grad[0,j]  + 0.0 for j in range(model.hid_size)])
            training_info["asc_k_grads"].append([model.neuron_layer.ln_asc_k.grad[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_amp_grads"].append([model.neuron_layer.asc_amp.grad[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_r_grads"].append([model.neuron_layer.asc_r.grad[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            # training_info["weight_grads"].append([torch.mean(model.neuron_layer.weight_iv.grad[:,m])  + 0.0 for m in range(model.hid_size)])

    
    if task == "pattern":
        final_outputs = []
        model.eval()
        for batch_ndx, sample in enumerate(init_dataloader):
            inputs, targets = sample
            _, nsteps, _ = inputs.shape
            model.reset_state()
            final_outputs.append(model.forward(inputs)[:, -nsteps:, :])
            plt.plot(model.forward(inputs)[0, -nsteps:, 0].detach().numpy())
            plt.plot(targets[0, -nsteps:, 0].detach().numpy())
            plt.close()
        training_info["final_outputs"] = final_outputs

        final_outputs_driven = []
        for batch_ndx, sample in enumerate(init_dataloader):
            inputs, targets = sample
            _, nsteps, _ = inputs.shape
            model.reset_state(batch_size = 1)

            # print(batch_ndx)

            # model(targets) # Predrive first dimension is batch so 1
            # model(targets_pre) # Predrive
            final_outputs_driven.append(model.forward(inputs[:, -nsteps:, :]))
        training_info["final_outputs_driven"] = final_outputs_driven
    elif task == "pattern_multid":
        final_outputs = []
        for batch_ndx, sample in enumerate(init_dataloader):
            inputs, targets = sample
            _, nsteps, _ = inputs.shape
            model.reset_state()
            outputs = model.forward(inputs)
            for dim in range(outputs.shape[-1]):
                final_outputs.append(outputs[:,:,dim])
        training_info["final_outputs"] = final_outputs
    return training_info

def km_reg(rbnn, reg_lambda):
	return reg_lambda * torch.mean((torch.exp(rbnn.neuron_layer.ln_k_m)) ** 2)

def aa_reg(rbnn, reg_lambda):
	return reg_lambda * (torch.mean(torch.exp(rbnn.neuron_layer.ln_k_m) ** 2) + torch.mean(torch.exp(rbnn.neuron_layer.ln_asc_k) ** 2) + torch.mean(rbnn.neuron_layer.asc_r ** 2) + torch.mean(rbnn.neuron_layer.asc_amp ** 2))


def train_rbnn_hfo(model, traindataset, batch_size, num_epochs, lr, reg_lambda, verbose = True, predrive = True, glifr = True, task = "pattern"):
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
    glifr : boolean, optional, default True
        whether to expect fields existing in GLIFR class
    
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
    if task == "copy":
        with torch.no_grad():
            data_gen = traindataset()
            traindataset = next(data_gen)

    training_info = {"losses": [],
                    "weights": [],
                    "threshes": [],
                    "k_ms": [],
                    "asc_amps": [],
                    "asc_rs": [],
                    "asc_ks": [],
                    "weight_grads": [],
                    "thresh_grads": [],
                    "k_m_grads": [],
                    "asc_amp_grads": [],
                    "asc_r_grads": [],
                    "asc_k_grads": []}
    model.eval()
    if task == "pattern":
        init_dataloader = tud.DataLoader(traindataset, batch_size=1, shuffle=False)
        init_outputs = []
        for batch_ndx, sample in enumerate(init_dataloader):
            inputs, targets = sample
            _, nsteps, _ = inputs.shape
            model.reset_state()
            init_outputs.append(model.forward(inputs)[:, -nsteps:, :])
        training_info["init_outputs"] = init_outputs

        init_outputs_driven = []
        for batch_ndx, sample in enumerate(init_dataloader):
            inputs, targets = sample
            _, nsteps, _ = inputs.shape
            model.reset_state()

            # model(targets) # Predrive
            # model(targets_pre) # Predrive
            init_outputs_driven.append(model.forward(inputs)[:, -nsteps:, :])
        training_info["init_outputs_driven"] = init_outputs_driven

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=1)
    optimizer = HessianFree(model.parameters())
    loss_fn = nn.MSELoss()
    # loss_fn = nn.SmoothL1Loss()
    trainloader = tud.DataLoader(traindataset, batch_size = batch_size, shuffle = True)
    model.train()
    # for batch_ndx, sample in enumerate(trainloader):
    for epoch in range(num_epochs):
        # if epoch % 1 == 0:
        #     trainloader = tud.DataLoader(traindataset, batch_size = batch_size, shuffle = True)
        if task == "copy":
            with torch.no_grad():
                traindataset = next(data_gen)
                trainloader = tud.DataLoader(traindataset, batch_size, shuffle = True)
        tot_loss = 0
        tot_pairs = 0
        loss_batch = []
        # if epoch % 10 == 0:
        #     print(model.neuron_layer.ln_asc_k)
        #     print(model.neuron_layer.ln_asc_k.grad) # -4 and -1 isn't good..-1 and 0 isn't good
        # for epoch in range(num_epochs):
        reg_lambda = 0.01
        for batch_ndx, sample in enumerate(trainloader):
            print(torch.mean(model.neuron_layer.weight_ih))
            # print(batch_ndx)
            n_subiter = 1
            for i in range(n_subiter):
                loss = 0.0
                inputs, targets = sample
                # plt.plot(inputs[0,:,0].detach().numpy())
                # plt.plot(targets[0,:,0].detach().numpy())
                # plt.show()
                # inputs = inputs.detach()
                # targets = targets.detach()

                _, nsteps, _ = inputs.shape
                # with torch.no_grad():
                #     for i in range(10):
                #         a = random.randint(0, nsteps - 5)
                #         inputs[:, a:a+5, :] = 0
                tot_pairs += len(targets)

                if True:#batch_ndx == 0:
                    model.reset_state(len(targets))
                optimizer.zero_grad()

                def closure():
                    optimizer.zero_grad()
                    model.reset_state(len(targets))
                    loss = 0.0
                    outputs = model(inputs)
                    loss = loss + loss_fn(outputs, targets)
                    if glifr:
                        loss = loss + aa_reg(model, reg_lambda = 0.008)
                    return (loss, outputs)

                if task == "copy" and epoch == num_epochs - 1:
                    np.savetxt('outputs_rnn_1000lng.txt', torch.squeeze(outputs).detach().numpy(), fmt="%i")
                    np.savetxt('targets_rnn_1000lng.txt', torch.squeeze(targets).detach().numpy(), fmt="%i")
                    np.savetxt('inputs_rnn_1000lng.txt', torch.squeeze(inputs).detach().numpy(), fmt="%i")

                
                optimizer.step(closure)
                model.reset_state(len(targets))
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                # if epoch % 2 == 0 and epoch < 20 and i % n_subiter == 0:
                # scheduler.step()
                
                tot_loss += loss.item()
                loss_batch.append(loss.item() / len(targets))
        if verbose:
            print(f"epoch {epoch}/{num_epochs}: loss of {tot_loss / tot_pairs} with variance {0 if len(loss_batch) < 2 else stat.variance(loss_batch)}")

        training_info["losses"].append(tot_loss)
        # training_info["weights"][0].append([model.input_linear.weight[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.in_size)])
        # if glifr:
        #     training_info["weights"][1].append([model.rec_linear.weight[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.hid_size)])
        # training_info["weights"][2].append([model.output_linear.weight[i,j].item() + 0.0 for i in range(model.out_size) for j in range(model.hid_size)])

        if glifr:
            training_info["k_ms"].append([torch.exp(model.neuron_layer.ln_k_m[0,j])  + 0.0 for j in range(model.hid_size)])
            training_info["threshes"].append([model.neuron_layer.thresh[0,j]  + 0.0 for j in range(model.hid_size)])
            training_info["asc_ks"].append([torch.exp(model.neuron_layer.ln_asc_k[j,0,m])  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_amps"].append([model.neuron_layer.asc_amp[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_rs"].append([model.neuron_layer.asc_r[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            # training_info["weights"].append([torch.mean(model.neuron_layer.weight_iv[:,m])  + 0.0 for m in range(model.hid_size)])

    final_outputs = []
    model.eval()
    for batch_ndx, sample in enumerate(init_dataloader):
        inputs, targets = sample
        _, nsteps, _ = inputs.shape
        model.reset_state()
        final_outputs.append(model.forward(inputs)[:, -nsteps:, :])
        plt.plot(model.forward(inputs)[0, -nsteps:, 0].detach().numpy())
        plt.plot(targets[0, -nsteps:, 0].detach().numpy())
        plt.show()
    training_info["final_outputs"] = final_outputs

    final_outputs_driven = []
    for batch_ndx, sample in enumerate(init_dataloader):
        inputs, targets = sample
        _, nsteps, _ = inputs.shape
        model.reset_state(batch_size = 1)

        print(batch_ndx)

        # model(targets) # Predrive first dimension is batch so 1
        # model(targets_pre) # Predrive
        final_outputs_driven.append(model.forward(inputs[:, -nsteps:, :]))
    training_info["final_outputs_driven"] = final_outputs_driven
    return training_info


class ThreeBitDataset(tud.Dataset):
    """
    creates a pytorch dataset for loading
    the data during training.
    """

    def __init__(self, sequence_length, dataset_length=1):
        self.sequence_length = sequence_length
        self.dataset_length = dataset_length

        self.make_trials()

    def __len__(self):
        return self.dataset_length

    def make_trials(self):
        """
        Generate the set of trials to be used fpr traomomg
        """
        seq_len = self.sequence_length

        self.items = {}
        for i in track(
            range(self.dataset_length * 2),
            description="Generating data...",
            total=self.dataset_length * 2,
            transient=True,
        ):
            X_batch = torch.zeros((seq_len, 3))
            Y_batch = torch.zeros((seq_len, 3))

            for m in range(3):
                # Define input
                X = torch.zeros(seq_len)
                Y = torch.zeros(seq_len)

                flips = (
                    rnd.uniform(1, seq_len - 1, int(seq_len / 200))
                ).astype(np.int32)
                flips2 = (
                    rnd.uniform(1, seq_len - 1, int(seq_len / 200))
                ).astype(np.int32)

                X[flips] = 1
                X[flips2] = -1
                X[0] = 1

                # Get correct output
                state = 0
                for n, x in enumerate(X):
                    if x == 1:
                        state = 1
                    elif x == -1:
                        state = -1

                    Y[n] = state

                # RNN input: batch size * seq len * n_input
                X = X.reshape(1, seq_len, 1)

                # out shape = (batch, seq_len, num_directions * hidden_size)
                Y = Y.reshape(1, seq_len, 1)

                X_batch[:, m] = X.squeeze()
                Y_batch[:, m] = Y.squeeze()

            self.items[i] = (X_batch, Y_batch)

    def __getitem__(self, item):
        X_batch, Y_batch = self.items[item]

        return X_batch, Y_batch


def make_batch(seq_len):
    """
    Return a single batch of given length
    """
    dataloader = torch.utils.data.DataLoader(
        ThreeBitDataset(seq_len, dataset_length=1),
        batch_size=1,
        num_workers=0,
        shuffle=True,
        worker_init_fn=lambda x: np.random.seed(),
    )

    batch = [b for b in dataloader][0]
    return batch


def plot_predictions(model, seq_len, batch_size):
    """
    Run the model on a single batch and plot
    the model's prediction's against the
    input data and labels.
    """
    X, Y = make_batch(seq_len)
    o = model.forward(X)

    f, axarr = plt.subplots(nrows=3, figsize=(12, 9))
    for n, ax in enumerate(axarr):
        ax.plot(X[0, :, n], lw=2, color=salmon, label="input")
        ax.plot(
            Y[0, :, n],
            lw=3,
            color=indigo_light,
            ls="--",
            label="correct output",
        )
        ax.plot(o.detach().numpy()[0, :, n], lw=2, color=light_green_dark, label="model output")
        ax.set(title=f"Input {n}")
        ax.legend()

    f.tight_layout()
    clean_axes(f)

def generate_poisson_noise_np(prob_pattern, freezing_seed=None):
    if isinstance(prob_pattern, list):
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

    shp = prob_pattern.shape

    if not(freezing_seed is None): rng = rnd.RandomState(freezing_seed)
    else: rng = rnd.RandomState()

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
    return spikes

def generate_click_task_data(batch_size, seq_len, n_neuron, recall_duration, p_group, f0=0.5,
                             n_cues=7, t_cue=100, t_interval=150,
                             n_input_symbols=4):
    t_seq = seq_len
    n_channel = n_neuron // n_input_symbols

    # randomly assign group A and B
    prob_choices = np.array([p_group, 1 - p_group], dtype=np.float32)
    idx = rnd.choice([0, 1], batch_size)
    probs = np.zeros((batch_size, 2), dtype=np.float32)
    # assign input spike probabilities
    probs[:, 0] = prob_choices[idx]
    probs[:, 1] = prob_choices[1 - idx]

    cue_assignments = np.zeros((batch_size, n_cues), dtype=np.int)
    # for each example in batch, draw which cues are going to be active (left or right)
    for b in range(batch_size):
        cue_assignments[b, :] = rnd.choice([0, 1], n_cues, p=probs[b])

    # generate input nums - 0: left, 1: right, 2:recall, 3:background noise
    input_nums = 3*np.ones((batch_size, seq_len), dtype=np.int)
    input_nums[:, :n_cues] = cue_assignments
    input_nums[:, -1] = 2

    # generate input spikes
    input_spike_prob = np.zeros((batch_size, t_seq, n_neuron))
    d_silence = t_interval - t_cue
    for b in range(batch_size):
        for k in range(n_cues):
            # input channels only fire when they are selected (left or right)
            c = cue_assignments[b, k]
            # reverse order of cues
            #idx = sequence_length - int(recall_cue) - k - 1
            idx = k
            input_spike_prob[b, d_silence+idx*t_interval:d_silence+idx*t_interval+t_cue, c*n_channel:(c+1)*n_channel] = f0

    # recall cue
    input_spike_prob[:, -recall_duration:, 2*n_channel:3*n_channel] = f0
    # background noise
    input_spike_prob[:, :, 3*n_channel:] = f0/4.
    input_spikes = generate_poisson_noise_np(input_spike_prob)

    # generate targets
    target_mask = np.zeros((batch_size, seq_len), dtype=np.bool)
    target_mask[:, -1] = True
    target_nums = np.zeros((batch_size, seq_len), dtype=np.int)
    target_nums[:, :] = np.transpose(np.tile(np.sum(cue_assignments, axis=1) > int(n_cues/2), (seq_len, 1)))

    return input_spikes, input_nums, target_nums, target_mask



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
