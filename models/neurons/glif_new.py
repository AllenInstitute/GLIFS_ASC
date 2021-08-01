"""
@chloewin
This file defines models for single layers of neurons.
"""
import matplotlib.pyplot as plt
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import neurons.utils_glif_new as uts


class BNNC(nn.Module):
        """
        Defines a single layer of RGLIF-ASC neurons

        Parameters
        ----------
        input_size : int
                number of dimensions of input
        hidden_size : int
                number of neurons
        bias : boolean
                whether or not incoming signals should be biased
        dt : float
                duration of timestep
        """
        def __init__(self, input_size, hidden_size, bias = True, dt=0.05, initburst=False, ascs=True, learnparams=True, sparseness=0):
                super().__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.num_ascs = 2
                self.ascs = ascs
                self.learnparams = learnparams

                self.weight_iv = Parameter(torch.randn((input_size, hidden_size)))
                self.weight_lat = Parameter(torch.randn((hidden_size, hidden_size)))

                num_keep = int((1 - sparseness) * (hidden_size**2))
                self.weight_lat_mask = torch.from_numpy(np.array([0] * (hidden_size ** 2 - num_keep) + [1] * num_keep)).reshape(self.weight_lat.shape)
                
                # self.c_m_inv = 0.02
                self.thresh = Parameter(torch.ones((1, hidden_size), dtype=torch.float), requires_grad=True)
                trans_k_m = math.log(0.02 * dt / (1 - (0.02 * dt)))#math.log(.05)
                self.trans_k_m = Parameter(trans_k_m * torch.ones((1, hidden_size), dtype=torch.float), requires_grad=True)
                # asc_amp = (-1, 1)
                # asc_r = (1,-1)

                self.asc_amp = Parameter(torch.tensor((-1,1)).reshape((2, 1, 1)) * torch.ones((2,1,hidden_size), dtype=torch.float) + torch.randn((2, 1, hidden_size), dtype=torch.float)) #Parameter(torch.ones((self.num_ascs,1,hidden_size), dtype=torch.float), requires_grad=True)
                self.trans_asc_k = Parameter(math.log(2) * torch.ones((self.num_ascs, 1, hidden_size), dtype=torch.float), requires_grad=True)
                self.trans_asc_r = Parameter(torch.tensor((math.log((1-0.99) / (1+0.99)),math.log((1+0.99) / (1-0.99)))).reshape((2, 1, 1)) * torch.ones((2,1,hidden_size), dtype=torch.float) + torch.randn((2, 1, hidden_size), dtype=torch.float))#Parameter(torch.ones((self.num_ascs,1,hidden_size), dtype=torch.float), requires_grad=True)                
                
                if not initburst:
                    nn.init.normal_(self.trans_asc_r, 0, 0.01)
                    nn.init.normal_(self.asc_amp, 0, 0.01)
                self.v_reset = 0

                if not learnparams:
                    self.thresh.requires_grad = False
                    self.trans_k_m.requires_grad = False
                    self.asc_amp.requires_grad = False
                    self.trans_asc_k.requires_grad = False
                    self.trans_asc_r.requires_grad = False

                self.sigma_v = 1
                self.gamma = 1
                self.dt = dt

                self.R = 0.1
                self.I0 = 0

                # randomly initializes incoming weights
                with torch.no_grad():
                        nn.init.uniform_(self.weight_iv, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))

        def spike_fn(self, x):
                """
                Propagates input through spiking activation function.
                
                Parameters
                ----------
                x : Tensor(any size)
                        input to spiking function
                
                Returns
                -------
                Tensor(same size as x)
                        tanh(x)
                """
                activation = self.gamma * (x - self.thresh) / self.sigma_v
                return torch.sigmoid(activation)
        
        def forward(self, x, firing, voltage, ascurrent, syncurrent, firing_delayed=None):
                """
                Propagates spike forward
                
                Parameters
                ----------
                x : torch tensor (n, ndims)
                        n inputs each with ndims dims
                firing : torch tensor (n, ndims)
                        previous firing rate
                voltage : torch tensor (n, ndims)
                        previous voltage
                ascurrent : torch tensor (n, ndims)
                        previous ascurrent
                syncurrent : torch tensor (n, ndims)
                        previous syncurrent
                """
                with torch.no_grad():
                    self.weight_lat.data = torch.mul(self.weight_lat.data, self.weight_lat_mask)
                if firing_delayed is None:
                    firing_delayed = copy(firing)
                # 1.5, -0.5 for lnasck
                """print(f"x: {x.shape}")
                print(f"wt: {self.weight_iv.shape}")
                print(f"fd: {firing_delayed.shape}")
                print(f"wl: {self.weight_lat.shape}")
                quit()"""
                syncurrent = x @ self.weight_iv + firing_delayed @ self.weight_lat
                
                if self.ascs:
                    ascurrent = (ascurrent * self.transform_to_asc_r(self.trans_asc_r) + self.asc_amp) * firing + (1 - self.dt * self.transform_to_k(self.trans_asc_k)) * ascurrent
                
                voltage = syncurrent + self.dt * self.transform_to_k(self.trans_k_m) * self.R * (torch.sum(ascurrent, dim=0) + self.I0) + (1 - self.dt * self.transform_to_k(self.trans_k_m)) * voltage - firing * (voltage - self.v_reset)
                firing = self.spike_fn(voltage)
                return firing, voltage, ascurrent, syncurrent
        
        def transform_to_asc_r(self, param):
                return 1 - (2 * torch.sigmoid(param)) # training on ln((1-r_j)/(1+r_j))
        
        def transform_to_k(self, param):
                return torch.sigmoid(param) / self.dt

class RNNC(nn.Module): # The true RNNC
        """
        Defines single recurrent layer

        Parameters
        ----------
        input_size : int
                number of dimensions in input
        hidden_size : int
                number of neurons
        bias : boolean
                whether bias should be used
        """
        def __init__(self, input_size, hidden_size, bias = True, sparseness=0):
                super().__init__()
                self.weight_ih = Parameter(torch.randn((input_size, hidden_size)))
                self.weight_lat = Parameter(torch.randn((hidden_size, hidden_size)))
                num_keep = int((1 - sparseness) * (hidden_size**2))
                self.weight_lat_mask = torch.from_numpy(np.array([0] * (hidden_size ** 2 - num_keep) + [1] * num_keep)).reshape(self.weight_lat.shape)

                if bias:
                        self.bias = Parameter(torch.zeros((1, hidden_size)))
                else:
                        self.bias = torch.zeros((1, hidden_size))

                with torch.no_grad():
                        nn.init.normal_(self.weight_ih, 0, 1 / math.sqrt(hidden_size))
                        nn.init.normal_(self.weight_lat, 0, 1 / math.sqrt(hidden_size))


        def forward(self, x, hidden, hidden_delayed):
                """
                Propagates single timestep
                
                Parameters
                ----------
                x : torch tensor (n, ndim)
                        input signal
                hidden : torch tensor (n, ndim)
                        previous hidden state
                
                Return
                ------
                """
                with torch.no_grad():
                    self.weight_lat.data = torch.mul(self.weight_lat.data, self.weight_lat_mask)
                hidden = torch.mm(x, self.weight_ih) + torch.mm(hidden_delayed, self.weight_lat) + self.bias
                hidden = torch.tanh(hidden)
                return hidden
