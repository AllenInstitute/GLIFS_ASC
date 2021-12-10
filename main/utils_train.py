"""
This file contains utils functions related to training.
"""

from torch.utils.data.dataset import TensorDataset

import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
import torch.utils.data as tud
from torchvision import datasets, transforms

def mnist_generator(root, batch_size):
    """
    Creates training and testing dataloader for MNIST

    Arguments
    ---------
    root : str
        filepath to MNIST dataset
    batch_size : int
        batch size to use for training and testing dataloaders
    """
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
    """from torchvision import datasets
    new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    datasets.MNIST.resources = [
    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
    for url, md5 in datasets.MNIST.resources
    ]
    """
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
    
def train_rbnn_mnist(model, batch_size, num_epochs, lr, glifr, verbose = True, linebyline=True, trainparams=True, ascs=True, sgd=False, output_text_filename="results.txt", trainloader = None, testloader = None, anneal=False):
    """
    Train RBNN model using trainloader and track metrics.
    Parameters
    ----------
    model : 
        network to be trained
    batch_size : int
        size of batches to be used during training
    num_epochs : int
        number of epochs to train the model for
    lr : float
        learning rate to use
    glifr : boolean, optional, default True
        whether to expect fields existing in GLIFR class
    verbose : boolean, optional, default True
        whether to print training loss over epochs and other information
    linebyline : boolean, optional, default True
        whether MNIST images should be read line by line
        as opposed to pixel by pixel
    trainparams : boolean, optional, default True
        whether to expect .grad on neuronal parameters
    ascs : boolean, optional, default True
        whether to expect trainable parameters relating to after-spike
        currents
    sgd : boolean, optional, default False
        whether to use stochastic gradient descent optimizer as opposed
        to Adam optimizer
    output_text_filename : string, optional, default "results.txt"
        name of file to print testing accuracy and loss to
    trainloader : Dataloader, optional, default None
        training loader to use; if None, default MNIST training data
        is used
    testloader : Dataloader, optional, default None
        training loader to use; if None, default MNIST training data
        is used
    anneal: boolean, optional, default False
        whether to reduce sigma_v over training epochs
    
    Returns
    -------
    training_info : Dict[Str: Any]
        dictionary including following information
        - losses : List[float]
            training loss over epochs
        - weights : List[List[float]]
            list of average weights incoming to each hidden neuron
            at each epoch
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
        - asc_k_grads : List[List[float]]
            ith element is a list of asc_r gradients of neurons at
            the ith epoch
        - test_accuracy : float
            final testing accuracy achieved by trained model
    """
    print(f"training with glifr {glifr}, linebyline {linebyline}, trainparams {trainparams}, ascs {ascs}, anneal {anneal}, and sgd {sgd}")

    training_info = {"losses": [],
                    "weights": [],
                    "threshes": [],
                    "k_ms": [],
                    "asc_amps": [],
                    "asc_rs": [],
                    "asc_ks": [],
                    "thresh_grads": [],
                    "k_m_grads": [],
                    "asc_amp_grads": [],
                    "asc_r_grads": [],
                    "asc_k_grads": []}

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    if sgd:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    sigma_vs = np.linspace(1e-3, 1, num=num_epochs)
    sigma_vs[0:int(num_epochs/2)] = 1e-3
    sigma_vs[int(num_epochs/2):-1] = 1
    if trainloader is None:
        root = './data/mnist'
        trainloader, testloader = mnist_generator(root, batch_size)

    model.train()
    for epoch in range(num_epochs):
        if glifr and anneal:
            with torch.no_grad():
                model.neuron_layer.sigma_v = sigma_vs[-1 - epoch]
        tot_loss = 0
        tot_pairs = 0
        loss_batch = []

        for batch_ndx, (data,target) in enumerate(trainloader):
            target = target.long()

            n_subiter = 1
            if batch_ndx % 100 == 0 and batch_ndx > 0:
                print(f"loss of {loss_batch[-1]} on batch {batch_ndx}/{len(trainloader)}")
            for i in range(n_subiter):
                loss = 0.0
                if linebyline:
                    data = data.view(-1, 28, 28)
                else:
                    data = data.view(-1, 28 * 28, 1)
                
                model.reset_state(len(target))
                optimizer.zero_grad()

                outputs = model(data)
                if linebyline:
                    outputs = torch.swapaxes(outputs, 1, 2)[:,:,-1]
                else:
                    outputs = outputs.reshape(len(target), 10, 28 * 28)[:,:,-1]

                loss = loss + loss_fn(outputs, target)

                loss.backward()    
                optimizer.step()
                
                tot_loss += loss.item()
                tot_pairs += len(target)
                loss_batch.append(loss.item() / len(target))
        if verbose:
            print(f"epoch {epoch}/{num_epochs}: loss of {tot_loss / tot_pairs} with variance {0 if len(loss_batch) < 2 else stat.variance(loss_batch)}")

        training_info["losses"].append(tot_loss)
        if glifr:
            training_info["k_ms"].append([model.neuron_layer.transform_to_k(model.neuron_layer.trans_k_m[0,j]).item()  + 0.0 for j in range(model.hid_size)])
            training_info["threshes"].append([model.neuron_layer.thresh[0,j].item()  + 0.0 for j in range(model.hid_size)])
            training_info["asc_ks"].append([model.neuron_layer.transform_to_k(model.neuron_layer.trans_asc_k[j,0,m]).item()  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_amps"].append([model.neuron_layer.asc_amp[j,0,m].item()  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_rs"].append([model.neuron_layer.transform_to_asc_r(model.neuron_layer.trans_asc_r)[j,0,m].item()  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])

            if epoch % 10 == 0 and trainparams and ascs:
                training_info["k_m_grads"].append([model.neuron_layer.trans_k_m.grad[0,j].item()  + 0.0 for j in range(model.hid_size)])
                training_info["thresh_grads"].append([model.neuron_layer.thresh.grad[0,j].item()  + 0.0 for j in range(model.hid_size)])
                training_info["asc_k_grads"].append([model.neuron_layer.trans_asc_k.grad[j,0,m].item()  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
                training_info["asc_amp_grads"].append([model.neuron_layer.asc_amp.grad[j,0,m].item()  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
                training_info["asc_r_grads"].append([model.neuron_layer.trans_asc_r.grad[j,0,m].item()  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])

    model.eval()
    
    # Test model on testloader
    test_loss = 0
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for data, target in testloader:
            target = target.long()
            model.reset_state(len(target))
            num_samples += len(target)

            if linebyline:
                data = data.view(-1, 28, 28)
            else:
                data = data.view(-1, 28 * 28, 1)

            output = model(data)
            if linebyline:
                output = torch.swapaxes(output, 1, 2)[:,:,-1]
            else:
                output = output.reshape(len(target), 10, 28 * 28)[:,:,-1]
            test_loss += loss_fn(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            
            correct += (pred == ((target.data.view_as(pred)))).sum()
    test_loss = test_loss * 1.0 / len(testloader.dataset)
    
    print(f"testing loss: {test_loss}")
    print(f"testing accuracy: {correct * 1.0 / num_samples}")

    training_info["test_accuracy"] = correct * 1.0 / num_samples

    original_stdout = sys.stdout
    with open(output_text_filename, 'w') as f:
        sys.stdout = f
        print(f"loss: {test_loss}")
        print(f"accuracy: {correct * 1.0 / len(testloader.dataset)}")
        sys.stdout = original_stdout

    return training_info

def train_rbnn_pattern(model, traindataset, batch_size, num_epochs, lr, verbose = True, glifr = True, trainparams=True, ascs=True, decay=False, sgd=False):
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
    verbose : boolean, optional, default True
        whether to print loss
    glifr : boolean, optional, default True
        whether to expect fields existing in GLIFR class
    trainparams : boolean, optional, default True
        whether to expect .grad on neuronal parameters
    ascs : boolean, optional, default True
        whether to expect trainable parameters relating to after-spike
        currents
    sgd : boolean, optional, default False
        whether to use stochastic gradient descent optimizer as opposed
        to Adam optimizer
    decay : boolean, optional, default False
        whether learning rate should be decaysed over training with 
        exponential learning rate scheduler (gamma=0.999)
    
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
        - weights : List[List[float]]
            list of average weights incoming to each hidden neuron
            at each epoch
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
        - asc_k_grads : List[List[float]]
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
                    "thresh_grads": [],
                    "k_m_grads": [],
                    "asc_amp_grads": [],
                    "asc_r_grads": [],
                    "asc_k_grads": []}
    model.eval()
    init_dataloader = tud.DataLoader(traindataset, batch_size=1, shuffle=False)
    init_outputs = []
    for _, sample in enumerate(init_dataloader):
        inputs, targets = sample
        _, nsteps, _ = inputs.shape
        model.reset_state()
        init_outputs.append(model.forward(inputs)[:, -nsteps:, :])
    training_info["init_outputs"] = init_outputs
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    if sgd:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)
    loss_fn = nn.MSELoss()
    trainloader = tud.DataLoader(traindataset, batch_size = batch_size, shuffle = True)
    model.train()
    for epoch in range(num_epochs):
        tot_loss = 0
        tot_pairs = 0
        loss_batch = []

        for _, sample in enumerate(trainloader):
            loss = 0.0
            inputs, targets = sample
            
            _, nsteps, _ = inputs.shape
            tot_pairs += len(targets)

            model.reset_state(len(targets))
            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs[:, -nsteps:, :]
                
            loss = loss + loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            if decay:
                scheduler.step()
            
            tot_loss += loss.item()
            loss_batch.append(loss.item() / len(targets))
        if verbose:
            print(f"epoch {epoch}/{num_epochs}: loss of {tot_loss / tot_pairs} with variance {0 if len(loss_batch) < 2 else stat.variance(loss_batch)}")

        training_info["losses"].append(tot_loss)
   
        if glifr and trainparams:
            training_info["k_ms"].append([model.neuron_layer.transform_to_k(model.neuron_layer.trans_k_m[0,j])  + 0.0 for j in range(model.hid_size)])
            training_info["threshes"].append([model.neuron_layer.thresh[0,j].item()  + 0.0 for j in range(model.hid_size)])
            training_info["asc_ks"].append([model.neuron_layer.transform_to_k(model.neuron_layer.trans_asc_k[j,0,m]).item()  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_amps"].append([model.neuron_layer.asc_amp[j,0,m].item()  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_rs"].append([model.neuron_layer.transform_to_asc_r(model.neuron_layer.trans_asc_r)[j,0,m].item()  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["weights"].append([torch.mean(model.neuron_layer.weight_iv[:,m]).item()  + 0.0 for m in range(model.hid_size)])
            if epoch % 10 == 0:
                training_info["k_m_grads"].append([model.neuron_layer.trans_k_m.grad[0,j].item()  + 0.0 for j in range(model.hid_size)])
                training_info["thresh_grads"].append([model.neuron_layer.thresh.grad[0,j].item()  + 0.0 for j in range(model.hid_size)])
                if ascs:
                    training_info["asc_k_grads"].append([model.neuron_layer.trans_asc_k.grad[j,0,m].item()  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
                    training_info["asc_amp_grads"].append([model.neuron_layer.asc_amp.grad[j,0,m].item()  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
                    training_info["asc_r_grads"].append([model.neuron_layer.trans_asc_r.grad[j,0,m].item()  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])

    with torch.no_grad():
        final_outputs = []
        model.eval()
        loss = 0
        for _, sample in enumerate(init_dataloader):
            inputs, targets = sample
            _, nsteps, _ = inputs.shape
            model.reset_state()
            outputs = model.forward(inputs)[:, -nsteps:, :]
            final_outputs.append(outputs)

            loss = loss + loss_fn(outputs, targets).item()
            plt.plot(model.forward(inputs)[0, -nsteps:, 0].detach().numpy())
            plt.plot(targets[0, -nsteps:, 0].detach().numpy())
            plt.close()
        training_info["test_loss"] = loss
        training_info["final_outputs"] = final_outputs
        
    return training_info
