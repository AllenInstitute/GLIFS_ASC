# GLIFR
Generalized-leaky-integrate-and-firing-rate (GLIFR) models

Goal: To create a rate-based neuronal model that incorporates after-spike currents, to connect these in networks, and to train the model on tasks, learning neuronal parameters (e.g., threshold, after-spike current related terms) in addition to synaptic weights.

## Structure of codebase:
- main: contains all code
- main/models: contains all modeling code
- main/results: contains plotting code and results presented in paper

## Important files:
- main/models/neurons: defines a GLIFR class for the rate based neuronal model, and a RNNC class for a vanilla RNN cell
- main/models/networks: defines a neural network using GLIFRs, a neural network using RNNCs, and a neural network using LSTMCell
- main/utils_train: contains utils functions that demonstrate how to train networks
- main/pattern_generation.py: program for testing sinusoid generation
- main/smnist.py: program for testing SMNIST

## Usage
The GLIFR class enables creation and usage of a differentiable layer of neurons that express biological dynamics including after-spike currents in a rate-based manner. One may optimize the neuron's intrinsic parameters (i.e., V_th, k_m, a_j, r_j, k_j) using standard gradient descent.

```
from models.neurons import GLIFR
neuron = GLIFR(input_size, hidden_size, num_ascs, hetinit, ascs, learnparams)
inputs = torch.randn(1, nsteps, input_size)

firing = torch.zeros((1, hidden_size))
voltage = torch.zeros((1, hidden_size))
syncurrent = torch.zeros((1, hidden_size))
ascurrents = torch.zeros((num_ascs, 1, hidden_size))

outputs = [torch.zeros((1, hidden_size)) for i in range(delay)]
for i in range(nsteps):
  x = inputs[:, i, :]
  firing, voltage, ascurrents, syncurrent = neuron(x, firing, voltage, ascurrents, syncurrent, outputs_[-delay]) # Set firing_delayed (i.e., last argument) to incorporate synaptic delay
  outputs.append(firing.clone())
```
- ```input_size```: the number of inputs the layer should expect
- ```hidden_size```: the number of neurons in the layer
- ```num_ascs```: the number of after-spike currents to be modeled
- ```het_init```: whether the intrinsic neuronal parameters should be initialized heterogeneously
- ```ascs```: whether after-spike currents should be modeled
- ```learnparams```: whether neuronal parameters should retain gradients

The BNNFC class enables creation and usage of a single layer network of GLIF neurons in which inputs are weighted, propagated through a single layer of GLIFR neurons, and finally weighted.

```
from models.networks import BNNFC
net = BNNFC(in_size, hid_size, out_size, dt, hetinit, ascs, learnparams)
inputs = torch.randn(1, nsteps, input_size)
outputs = neuron(inputs)
```
- ```in_size```: the number of inputs the network receives
- ```hidden_size```: the number of neurons in the hidden layer
- ```num_ascs```: the number of after-spike currents to be modeled
- ```het_init```: whether the intrinsic neuronal parameters should be initialized heterogeneously
- ```ascs```: whether after-spike currents should be modeled
- ```learnparams```: whether neuronal parameters should retain gradients

Both smnist_results and pattern_results use command line arguments to run training simulations on the RNN, BNNFC, and LSTM networks.
