# Neuronal Heterogeneity Learned by Gradient Descent Improves Performance on Temporally Challenging Tasks
This repository contains code for the models and analysis used in the paper, "Neuronal Heterogeneity Learned by Gradient Descent Improves Performance on Temporally Challenging Tasks."

We develop a differentiable rate-based neuronal model that incorporates after-spike currents and term it the "generalized-leaky-integrate-and-firing-rate" model. We create networks using this model and train them on tasks, optimizing both neuronal parameters underlying intrinsic dynamics (e.g., threshold, membrane time constant) and synaptic weights.

## Structure of codebase:
- main: contains all code
- main/config: contains configuration files for examples
- main/data: folder to store data for training/testing
- main/datasets: code to process training/testing data
- main/examples: all code used for training/testing described in paper, including plotting code for generating figures
- main/figures: supplementary figures and figures used in paper
- main/models: main code for model utilization
- main/training: utils functions for training
- main/utils: other basic utils functions

## Important files:
- main/models/neurons.py: defines a GLIFR class for the rate based neuronal model, and a RNNC class for a vanilla RNN cell
- main/models/pl_modules.py: defines a neural network using GLIFRs, a neural network using RNNCs, and a neural network using LSTMCell
- main/utils/types.py: defines named tuples used as parameters in model code
- main/models/analyzer.py: defines an analyzer class to run certain analyses on trained models
- main/examples/train.py: main training code used in paper

## Usage
The GLIFR class enables creation and usage of a differentiable layer of neurons that express biological dynamics including after-spike currents in a rate-based manner. One may optimize the neuron's intrinsic parameters (i.e., V_th, k_m, a_j, r_j, k_j) using standard gradient descent.

```
from models.neurons import GLIFR
from utils.types import StructureParameters, NeuronParameters

structure_parameters = StructureParameters(input_size, hidden_size, output_size)
neuron_parameters = NeuronParameters(dt, tau)
neuron = GLIFR(structure_parameters, neuron_parameters)

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

The GLIFRN class enables creation and usage of a single layer network of GLIF neurons in which inputs are weighted, propagated through a single layer of GLIFR neurons, and finally weighted. This class can also be used with PytorchLightning.

```
from models.pl_modules import GLIFRN
parser = ArgumentParser()
add_structure_args(parser)
add_general_model_args(parser)
GLIFRN.add_model_specific_args(parser)
args = parser.parse_args()

net = GLIFRN(**vars(args))
inputs = torch.randn(1, nsteps, input_size)
outputs = neuron(inputs)
```

The file examples/train.py use command line arguments and the config files in main/config, which can be edited by the user, to run training simulations on the RNN, GLIFRN, and LSTM networks. The MNIST related tasks assume a data directory containing the MNIST dataset in standard format. Below are some example commands.

- python3 train.py --task lmnist --ntrials 10 --num_workers 8 --accelerator gpu --gpus 2 --strategy ddp_find_unused_parameters_false # Runs 10 trials of LMNIST training using 8 workers on the GPU (using 2 devices)
- python3 train.py --task sine --ntrials 10 # Runs 10 trails of sine training on the CPU

The results we obtained are stored in main/examples/results, and the plotting and analysis code (main/examples/results/analysis) may be run with that.