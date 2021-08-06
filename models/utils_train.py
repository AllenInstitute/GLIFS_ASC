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
    
def train_rbnn_mnist(model, batch_size, num_epochs, lr, glifr, verbose = True, linebyline=True, trainparams=True, ascs=True, sgd=False, output_text_filename="results.txt"):#, batch_size, num_epochs, lr, reg_lambda, verbose = True, predrive = True, glifr = True, task = "pattern"):
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
    if sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    root = './data/mnist'
    trainloader, testloader = mnist_generator(root, batch_size)
    model.train()

    if linebyline:
        input_channels = 28
    else:
        input_channels = 1
    seq_length = int(784/input_channels)
    # for batch_ndx, sample in enumerate(trainloader):
    for epoch in range(num_epochs):
        tot_loss = 0
        tot_pairs = 0
        loss_batch = []
        reg_lambda = 0.01
        for batch_ndx, (data,target) in enumerate(trainloader):
            target = target.long()

            n_subiter = 1
            # if batch_ndx % 100 == 0 and batch_ndx > 0:
            #     print(f"loss of {loss_batch[-1]} on batch {batch_ndx}/{len(trainloader)}")
            for i in range(n_subiter):
                loss = 0.0
                # print(data.shape)
                if linebyline:
                    data = data.view(-1, 28, 28)
                else:
                    data = data.view(-1, 28 * 28, 1)

                #optimizer.zero_grad()

                _, nsteps, _ = data.shape
                # with torch.no_grad():
                #     for i in range(10):
                #         a = random.randint(0, nsteps - 5)
                #         inputs[:, a:a+5, :] = 0
                tot_pairs += len(target)

                model.reset_state(len(target))
                optimizer.zero_grad()

                outputs = model(data)
                if linebyline:
                    outputs = outputs.reshape(len(target), 10, 28)[:,:,-1]
                else:
                    outputs = outputs.reshape(len(target), 10, 28 * 28)[:,:,-1]
                # outputs = outputs.reshape(len(target), 10, 28)[:,:,-1]#torch.mean(outputs.reshape(len(target), 10, 28), -1)
                loss = loss + loss_fn(outputs, target)
                # if i % n_subiter == 0:
                #     print(loss.item() / len(targets))
                # if glifr:
                #     loss = loss + aa_reg(model, reg_lambda = reg_lambda)
                #     reg_lambda *= 0.9
                # if glifr:
                #     loss = loss + km_reg(model, reg_lambda)
                loss.backward()
    
                optimizer.step()
                
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
            training_info["k_ms"].append([model.neuron_layer.transform_to_k(model.neuron_layer.trans_k_m[0,j])  + 0.0 for j in range(model.hid_size)])
            training_info["threshes"].append([model.neuron_layer.thresh[0,j]  + 0.0 for j in range(model.hid_size)])
            training_info["asc_ks"].append([model.neuron_layer.transform_to_k(model.neuron_layer.trans_asc_k[j,0,m])  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_amps"].append([model.neuron_layer.asc_amp[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_rs"].append([model.neuron_layer.transform_to_asc_r(model.neuron_layer.trans_asc_r)[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            # training_info["weights"].append([torch.mean(model.neuron_layer.weight_iv[:,m])  + 0.0 for m in range(model.hid_size)])

        # TODO: Replace with real weight
        # training_info["weight_grads"][0].append([model.input_linear.weight.grad[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.in_size)])
        # if glifr:
        #     training_info["weight_grads"][1].append([model.rec_linear.weight.grad[i,j].item() + 0.0 for i in range(model.hid_size) for j in range(model.hid_size)])
        # training_info["weight_grads"][2].append([model.output_linear.weight.grad[i,j].item() + 0.0 for i in range(model.out_size) for j in range(model.hid_size)])
        if glifr and epoch % 10 == 0 and trainparams and ascs:
            training_info["k_m_grads"].append([model.neuron_layer.trans_k_m.grad[0,j]  + 0.0 for j in range(model.hid_size)])
            training_info["thresh_grads"].append([model.neuron_layer.thresh.grad[0,j]  + 0.0 for j in range(model.hid_size)])
            training_info["asc_k_grads"].append([model.neuron_layer.trans_asc_k.grad[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_amp_grads"].append([model.neuron_layer.asc_amp.grad[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
            training_info["asc_r_grads"].append([model.neuron_layer.trans_asc_r.grad[j,0,m]  + 0.0 for j in range(model.neuron_layer.num_ascs) for m in range(model.hid_size)])
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
            if linebyline:
                data = data.view(-1, 28, 28)
            else:
                data = data.view(-1, 28 * 28, 1)
            output = model(data)
            if linebyline:
                output = output.reshape(len(target), 10, 28)[:,:,-1]
            else:
                output = output.reshape(len(target), 10, 28 * 28)[:,:,-1]
            test_loss += loss_fn(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            
            correct += (pred == ((target.data.view_as(pred)))).sum()
    test_loss = test_loss * 1.0 / len(testloader.dataset)
    
    print(f"loss: {test_loss}")
    print(f"accuracy: {correct * 1.0 / len(testloader.dataset)}")

    training_info["test_accuracy"] = correct * 1.0 / len(testloader.dataset)

    original_stdout = sys.stdout # Save a reference to the original standard output

    with open(output_text_filename, 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(f"loss: {test_loss}")
        print(f"accuracy: {correct * 1.0 / len(testloader.dataset)}")
        sys.stdout = original_stdout

    return training_info