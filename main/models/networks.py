"""
This file contains classes for the different networks that we used.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from copy import copy

from models.neurons import RNNC, GLIFR

class BNNFC(nn.Module):
        """
        Defines a single recurrent layer network of GLIFR neurons with a synaptic delay.

        Parameters
        ----------
        in_size : int
                number of inputs
        hid_size : int
                number of neurons in hidden layer
        out_size : int
                number of outputs
        num_ascs : int, default 2
                number of after-spike currents to model
        dt : float, default 0.05
                duration of timestep in ms
        hetinit : boolean, default False
                whether neuronal parameters should be initialized with
                heterogeneity across the network
        ascs : boolean, default True
                whether after-spike currents should be modeled/learned
        learnparams : boolean, default True
                whether intrinsic parameters should be trained
        output_weight : boolean, default True
                whether outputs of hidden layer should be weighted
        dropout_prob : float, default 0
                probability of dropout
        """
        def __init__(self, in_size, hid_size, out_size, num_ascs=2, dt=0.05, hetinit=False, ascs=True, learnparams=True, output_weight=True, dropout_prob=0):
                super().__init__()

                self.output_linear = nn.Linear(in_features = hid_size, out_features = out_size, bias = True)
                self.neuron_layer = GLIFR(input_size = in_size, hidden_size = hid_size, num_ascs=num_ascs, hetinit=hetinit, ascs=ascs, learnparams=learnparams)
                self.dropout_layer = nn.Dropout(p=dropout_prob, inplace=False)

                self.in_size = in_size
                self.hid_size = hid_size
                self.out_size = out_size

                self.num_ascs = num_ascs
                self.output_weight = output_weight
                self.dt = dt
                self.delay = int(1 / self.dt)

                self.idx = []
                self.silence_mult = torch.eye(self.hid_size)

                self.reset_state()

        def forward(self, input, track=False):
                """
                Propagates input through network.

                Parameters
                ----------
                input : Tensor(batch_size, nsteps, in_size)
                        input signal to be input over time
                track : bool
                        whether voltages, after-spike currents, and synaptic currents
                        over time should be returned in addition to network output
                
                Returns
                -------
                outputs : Tensor(batch_size, nsteps, out_size)
                        output weighted firing rates of network
                voltages : Tensor(batch_size, nsteps, hid_size) [returned if track]
                        voltages of hidden layer tracked over time
                ascs : Tensor(num_ascs, batch_size, nsteps, hid_size) [returned if track]
                        after-spike currents tracked over time
                syns : Tensor(batch_size, nsteps, hid_size) [returned if track]
                        synaptic current tracked over time
                """
                _, nsteps, in_size = input.shape
                assert(in_size == self.in_size), f"input has {in_size} size but network accepts {self.in_size} inputs"

                delay = self.delay
                outputs = torch.empty((self.batch_size, nsteps, self.out_size))
                voltages = torch.empty((self.batch_size, nsteps, self.hid_size))
                ascs = torch.empty((self.num_ascs, self.batch_size, nsteps, self.hid_size))
                syns = torch.empty((self.batch_size, nsteps, self.hid_size))
                outputs_ = [torch.zeros((self.batch_size, self.hid_size)) for i in range(delay)]

                self.firing_over_time = torch.zeros((self.batch_size, nsteps, self.hid_size))
                
                for step in range(nsteps):
                        x = input[:, step, :]
                        
                        self.firing, self.voltage, self.ascurrents, self.syncurrent = self.neuron_layer(x, self.firing, self.voltage, self.ascurrents, self.syncurrent, outputs_[-delay])
                        self.firing = self.dropout_layer(self.firing)
                        self.firing = torch.matmul(self.firing, self.silence_mult)
                        if self.output_weight:
                                x = self.output_linear(self.firing)
                        else:
                                x = (self.firing)
                        self.firing_over_time[:, step, :] = self.firing.clone()
                        outputs[:, step, :] = x
                        if track:
                            voltages[:, step, :] = self.voltage.clone()
                            ascs[:, :, step, :] = self.ascurrents.clone()
                            syns[:, step, :] = self.syncurrent.clone()
                        outputs_.append(self.firing.clone())
                        
                        if len(outputs_) > delay:
                            outputs_ = outputs_[-delay:]
                if track:
                        return outputs, voltages, ascs, syns 
                else:
                        return outputs

        def reset_state(self, batch_size = 1, full_reset = True):
                """
                Resets internal state of network

                Parameters
                ----------
                batch_size : int
                        batch size to be used subsequently
                full_reset : boolean
                        whether internal states should be re-initialized
                """
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
                """
                Silences specific neurons ffrom this point forward.

                Parameters
                ----------
                idx : list
                        list of indices of neurons whose outputs should be
                        silenced
                """
                self.idx = idx
                self.silence_mult = torch.eye(self.hid_size)
                for i in self.idx:
                        self.silence_mult[i, i] = 0

class RNNFC(nn.Module):
        """
        Defines a single recurrent layer network with a delay of dt.

        Parameters
        ----------
        in_size : int
                number of inputs
        hid_size : int
                number of neurons in hidden layer
        out_size : int
                number of outputs
        dt : float, default 0.05
                duration of timestep in ms
        output_weight : boolean, default True
                whether outputs of hidden layer should be weighted
        dropout_prob : float, default 0
                probability of dropout
        """
        def __init__(self, in_size, hid_size, out_size, dt=0.05, output_weight=True, dropout_prob=0):
                super().__init__()

                self.output_linear = nn.Linear(in_features = hid_size, out_features = out_size, bias = True)
                self.neuron_layer = RNNC(input_size = in_size, hidden_size = hid_size, bias = True)
                self.dropout_layer = nn.Dropout(p=dropout_prob, inplace=False)

                self.in_size = in_size
                self.hid_size = hid_size
                self.out_size = out_size

                self.num_ascs = 2

                self.dt = dt
                self.delay = 1
                self.output_weight = output_weight

                self.reset_state()
                self.idx = []
                self.silence_mult = torch.eye(self.hid_size)

        def forward(self, input, track = False):
                """
                Propagates input through network.

                Parameters
                ----------
                input : Tensor(batch_size, nsteps, in_size)
                        input signal to be input over time
                track : bool
                        whether voltages, after-spike currents, and synaptic currents
                        over time should be returned in addition to network output
                
                Returns
                -------
                outputs : Tensor(batch_size, nsteps, out_size)
                        output weighted firing rates of network
                voltages : Tensor(batch_size, nsteps, hid_size) [returned if track]
                        pre-activation outputs of RNN over time
                """

                delay = self.delay

                _, nsteps, in_size = input.shape
                assert(in_size == self.in_size), f"input has {in_size} size but network accepts {self.in_size} inputs"

                outputs = torch.empty((self.batch_size, nsteps, self.out_size))
                outputs_ = [torch.zeros((self.batch_size, self.hid_size)) for i in range(delay)]

                self.firing_over_time = torch.zeros((self.batch_size, nsteps, self.hid_size))
                voltages = torch.empty((self.batch_size, nsteps, self.hid_size))
                for step in range(nsteps):
                        x = input[:, step, :]
                        
                        if track:
                                self.firing, voltage = self.neuron_layer(x, self.firing, outputs_[-delay], track)
                                voltages[:, step, :] = voltage
                        else:
                                self.firing = self.neuron_layer(x, self.firing, outputs_[-delay], track)
                        self.firing = self.dropout_layer(self.firing)
                        self.firing = torch.matmul(self.firing, self.silence_mult)
                        if self.output_weight:
                                x = self.output_linear(self.firing)
                        else:
                                x = copy(self.firing)
                        outputs[:, step, :] = x
                        self.firing_over_time[:, step, :] = self.firing.clone()

                        outputs_.append(self.firing.clone())
                        if len(outputs_) > delay:
                            outputs_ = outputs_[-delay:]
                if track:
                        return outputs, voltages
                return outputs

        def reset_state(self, batch_size = 1):
                """
                Resets internal state of network

                Parameters
                ----------
                batch_size : int
                        batch size to be used subsequently
                full_reset : boolean
                        whether internal states should be re-initialized
                """
                self.batch_size = batch_size

                self.firing = torch.zeros((self.batch_size, self.hid_size))

        def silence(self, idx):
                """
                Silences specific neurons ffrom this point forward.

                Parameters
                ----------
                idx : list
                        list of indices of neurons whose outputs should be
                        silenced
                """
                self.idx = idx
                self.silence_mult = torch.eye(self.hid_size)
                for i in self.idx:
                        self.silence_mult[i, i] = 0

class LSTMFC(nn.Module):
        """
        Defines a single layer LSTM network with a synaptic delay of dt.

        Parameters
        ----------
        in_size : int
                number of inputs
        hid_size : int
                number of neurons in hidden layer
        out_size : int
                number of outputs
        dt : float, default 0.05
                duration of timestep in ms
        output_weight : boolean, default True
                whether outputs of hidden layer should be weighted
        dropout_prob : float, default 0
                probability of dropout
        """
        def __init__(self, in_size, hid_size, out_size, dt=0.05, output_weight=True, dropout_prob=0):
                super().__init__()
                self.output_linear = nn.Linear(in_features = hid_size, out_features = out_size, bias = True)
                self.neuron_layer = nn.LSTMCell(input_size = in_size, hidden_size = hid_size, bias = True)
                self.dropout_layer = nn.Dropout(p=dropout_prob, inplace=False)

                self.in_size = in_size
                self.hid_size = hid_size
                self.out_size = out_size
                self.delay = 1
                self.idx = []
                self.silence_mult = torch.eye(self.hid_size)
                self.output_weight = output_weight

                self.reset_state()

        def forward(self, input, track = False):
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
                outputs_ = [torch.zeros((self.batch_size, self.hid_size)) for i in range(delay)]

                self.firing_over_time = torch.zeros((self.batch_size, nsteps, self.hid_size))

                for step in range(nsteps):
                        x = input[:, step, :]
                        self.h, self.c = self.neuron_layer(x, (self.h,self.c))
                        self.h = self.dropout_layer(self.h)

                        self.h = torch.matmul(self.h, self.silence_mult)
                        if self.output_weight:
                                x = self.output_linear(self.h)
                        else:
                                x = copy(self.h)
                        outputs[:, step, :] = x

                        self.firing_over_time[:, step, :] = self.h.clone()

                        outputs_.append(self.h.clone())
                        if len(outputs_) > delay:
                            outputs_ = outputs_[-delay:]
                return outputs

        def reset_state(self, batch_size = 1):
                """
                Resets internal state of network

                Parameters
                ----------
                batch_size : int
                        batch size to be used subsequently
                full_reset : boolean
                        whether internal states should be re-initialized
                """
                self.batch_size = batch_size

                self.c = torch.zeros((self.batch_size, self.hid_size))
                self.h = torch.zeros((self.batch_size, self.hid_size))

        def silence(self, idx):
                """
                Silences specific neurons ffrom this point forward.

                Parameters
                ----------
                idx : list
                        list of indices of neurons whose outputs should be
                        silenced
                """
                self.idx = idx
                self.silence_mult = torch.eye(self.hid_size)
                for i in self.idx:
                        self.silence_mult[i, i] = 0
