import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from copy import copy

from neurons.glif_new import RNNC, BNNC

class BNNFC(nn.Module):
        """
        Defines a single recurrent layer network with a synaptic delay.

        Parameters
        ----------
        in_size : int
                number of inputs
        hid_size : int
                number of neurons in hidden layer
        out_size : int
                number of outputs
        """
        def __init__(self, in_size, hid_size, out_size, dt=0.05, initburst=False):
                super().__init__()

                self.output_linear = nn.Linear(in_features = hid_size, out_features = out_size, bias = True)
                self.neuron_layer = BNNC(input_size = in_size, hidden_size = hid_size, bias = True, initburst=initburst)

                self.in_size = in_size
                self.hid_size = hid_size
                self.out_size = out_size

                self.num_ascs = self.neuron_layer.num_ascs
                self.dt = dt
                self.delay = int(1 / self.dt)

                self.idx = []

                self.reset_state()

        def forward(self, input, target=None):
                """
                Propagates input through network.

                Parameters
                ----------
                input : Tensor(batch_size, nsteps, 1, in_size)
                        input signal to be input over time
                """
                _, nsteps, in_size = input.shape
                assert(in_size == self.in_size), f"input has {in_size} size but network accepts {self.in_size} inputs"

                delay = self.delay
                outputs = torch.empty((self.batch_size, nsteps, self.out_size))
                outputs_ = [torch.zeros((self.batch_size, self.hid_size)) for i in range(delay)]
                
                for step in range(nsteps):
                        x = input[:, step, :]
                        # x = torch.cat((x, outputs_[-delay]), dim=-1)
                        
                        self.firing, self.voltage, self.ascurrents, self.syncurrent = self.neuron_layer(x, self.firing, self.voltage, self.ascurrents, self.syncurrent, outputs_[-delay])
                        # TODO: this cutting down throws breaks the graph so need to fix that :)
                        if len(self.idx) > 0:
                            self.firing[:, self.idx] = 0
                        x = self.output_linear(self.firing)
                        outputs[:, step, :] = x
                        self.last_output = x
                        outputs_.append(copy(self.firing))
                        
                        if len(outputs_) > delay:
                            outputs_ = outputs_[-delay:]
                return outputs

        def reset_state(self, batch_size = 1, full_reset = True):
                self.batch_size = batch_size

                self.last_output = torch.zeros((self.batch_size, self.out_size))
                self.voltages = []

                full_reset = True
                
                if full_reset:
                    self.firing = torch.zeros((self.batch_size, self.hid_size))
                    self.voltage = torch.zeros((self.batch_size, self.hid_size))
                    self.syncurrent = torch.zeros((self.batch_size, self.hid_size))
                    self.ascurrents = torch.zeros((self.num_ascs, self.batch_size, self.hid_size))
                else:
                    self.firing = self.firing.detach()
                    self.voltage = self.voltage.detach()
                    self.syncurrent = self.syncurrent.detach()
                    self.ascurrents = self.ascurrents.detach()
        
        def silence(self, idx):
                self.idx = idx

class RNNFC(nn.Module):
        """
        Defines a single recurrent layer network.

        Parameters
        ----------
        in_size : int
                number of inputs
        hid_size : int
                number of neurons in hidden layer
        out_size : int
                number of outputs
        """
        def __init__(self, in_size, hid_size, out_size, dt=0.05):
                super().__init__()

                self.output_linear = nn.Linear(in_features = hid_size, out_features = out_size, bias = True)
                self.neuron_layer = RNNC(input_size = in_size, hidden_size = hid_size, bias = True)

                self.in_size = in_size
                self.hid_size = hid_size
                self.out_size = out_size

                self.num_ascs = 2

                self.dt = dt
                self.delay = int(1 / self.dt)

                self.reset_state()
                self.idx = []

        def forward(self, input):
                """
                Propagates input through network.
                Each step is computed by passing input through RNN

                Parameters
                ----------
                input : Tensor(batch_size, nsteps, 1, in_size)
                        input signal to be input over time
                """

                delay = 1#TODO:self.delay

                _, nsteps, in_size = input.shape
                assert(in_size == self.in_size), f"input has {in_size} size but network accepts {self.in_size} inputs"

                # outputs = []
                outputs = torch.empty((self.batch_size, nsteps, self.out_size))
                outputs_ = [torch.zeros((self.batch_size, self.hid_size)) for i in range(delay)]

                for step in range(nsteps):
                        x = input[:, step, :]
                        #x = torch.cat((x, outputs_[-delay]), dim=-1)
                        
                        self.firing = self.neuron_layer(x, self.firing, outputs_[-delay])
                        if len(self.idx) > 0: # TODO: please fix so no bad error 
                            self.firing[:, self.idx] = 0
                        x = self.output_linear(self.firing)
                        outputs[:, step, :] = x

                        outputs_.append(copy(self.firing))
                        if len(outputs_) > delay:
                            outputs_ = outputs_[-delay:]
                return outputs

        def reset_state(self, batch_size = 1):
                self.batch_size = batch_size

                self.firing = torch.zeros((self.batch_size, self.hid_size))

        def silence(self, idx):
                self.idx = idx

class LSTMFC(nn.Module):
        """
        Defines a single recurrent layer network.

        Parameters
        ----------
        in_size : int
                number of inputs
        hid_size : int
                number of neurons in hidden layer
        out_size : int
                number of outputs
        """
        def __init__(self, in_size, hid_size, out_size):
                super().__init__()
                self.output_linear = nn.Linear(in_features = hid_size, out_features = out_size, bias = True)
                self.neuron_layer = nn.LSTMCell(input_size = in_size + out_size, hidden_size = hid_size, bias = True)#RNNC(input_size = in_size, hidden_size = hid_size, bias = True)

                self.in_size = in_size
                self.hid_size = hid_size
                self.out_size = out_size
                self.delay = int(1 / self.dt)

                self.reset_state()

        def forward(self, input):
                """
                Propagates input through network.

                Parameters
                ----------
                input : Tensor(batch_size, nsteps, 1, in_size)
                        input signal to be input over time
                """
                delay = self.delay
                _, nsteps, in_size = input.shape
                assert(in_size == self.in_size), f"input has {in_size} size but network accepts {self.in_size} inputs"

                outputs = torch.empty((self.batch_size, nsteps, self.out_size))
                outputs_ = [torch.zeros((self.batch_size, self.out_size)) for i in range(delay)]
                c = torch.zeros((self.batch_size, self.hid_size))
                h = torch.zeros((self.batch_size, self.hid_size))

                for step in range(nsteps):
                        x = input[:, step, :]
                        x = torch.cat((x, outputs_[-delay]), dim=-1)
                        h, c = self.neuron_layer(x, (h,c))
                        x = self.output_linear(h)
                        outputs[:, step, :] = x

                        outputs_.append(x)
                        if len(outputs_) > delay:
                            outputs_ = outputs_[-delay:]
                return outputs

        def reset_state(self, batch_size = 1):
                self.batch_size = batch_size
