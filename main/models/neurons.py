"""
This file defines models for single layers of neurons.
"""
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class GLIFR(nn.Module):
        """
        Defines a single layer of GLIFR-ASC neurons

        Parameters
        ----------
        input_size : int
                number of dimensions of input
        hidden_size : int
                number of neurons
        num_ascs : int
                number of after-spike currents to model
        dt : float, default 0.05
                duration of timestep
        tau : float, defaults to dt
                conceptually the number of times the voltage and threshold are compared
        hetinit : boolean, default False
                whether parameters should be heterogeneously initialized
        ascs : boolean, default True
                whether after-spike currents and related parameter gradients should be maintained
        learnparams : boolean, default True
                whether parameters should be learned
        """
        def __init__(self, input_size, hidden_size, num_ascs=2, dt=0.05, tau=None, hetinit=False, ascs=True, learnparams=True):
                super().__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size

                self.num_ascs = num_ascs
                self.tau = tau
                if self.tau is None:
                        self.tau = dt

                self.ascs = ascs
                self.learnparams = learnparams

                self.weight_iv = Parameter(torch.randn((input_size, hidden_size))) # incoming weights
                self.weight_lat = Parameter(torch.randn((hidden_size, hidden_size))) # lateral connections (i.e., among neurons in same layer)

                if hetinit:
                    self.thresh = Parameter(-1 + 2 * torch.rand((1, hidden_size), dtype=torch.float), requires_grad=True)
                    
                    k_m = 0.04 + 0.02 * torch.rand((1, hidden_size), dtype=torch.float)
                    self.trans_k_m = Parameter(torch.log((k_m * dt) / (1 - (k_m * dt))), requires_grad=True)

                    k_asc = 1.5 + torch.rand((self.num_ascs, 1, hidden_size), dtype=torch.float)
                    self.trans_asc_k = Parameter(torch.log((k_asc * dt) / (1 - (k_asc * dt))), requires_grad=True)

                    asc_r = -0.1 + 0.2 * torch.rand((self.num_ascs, 1, hidden_size), dtype=torch.float)
                    self.trans_asc_r = Parameter(torch.log((1 - asc_r) / (1 + asc_r)), requires_grad=True)

                    self.asc_amp = Parameter(-0.1 + 0.2 * torch.rand((self.num_ascs, 1, hidden_size), dtype=torch.float), requires_grad=True)
                else:
                    self.thresh = Parameter(torch.ones((1, hidden_size), dtype=torch.float), requires_grad=True)
                    
                    trans_k_m = math.log(0.05 * dt / (1 - (0.05 * dt)))
                    self.trans_k_m = Parameter(trans_k_m * torch.ones((1, hidden_size), dtype=torch.float), requires_grad=True)

                    self.trans_asc_k = Parameter(math.log(2 * dt / (1 - (2 * dt))) * torch.ones((self.num_ascs, 1, hidden_size), dtype=torch.float), requires_grad=True)

                    asc_r = -0.01 + 0.02 * torch.rand((self.num_ascs, 1, hidden_size), dtype=torch.float)
                    self.trans_asc_r = Parameter(torch.log((1 - asc_r) / (1 + asc_r)), requires_grad=True)

                    self.asc_amp = Parameter(-0.01 + 0.02 * torch.rand((self.num_ascs, 1, hidden_size), dtype=torch.float), requires_grad=True)

                self.v_reset = 0

                if not learnparams:
                    self.thresh.requires_grad = False
                    self.trans_k_m.requires_grad = False
                    self.asc_amp.requires_grad = False
                    self.trans_asc_k.requires_grad = False
                    self.trans_asc_r.requires_grad = False

                if not ascs:
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
                        nn.init.uniform_(self.weight_lat, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))

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
                        output of spiking function
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
                firing_delayed: torch tensor (n, ndims), default None (identical to firing)
                        firing rate to use when propagating lateral connections

                Returns
                -------
                firing : torch tensor (n, ndims)
                        new firing rate
                voltage : torch tensor (n, ndims)
                        new voltage
                ascurrent : torch tensor (n, ndims)
                        new ascurrent
                syncurrent : torch tensor (n, ndims)
                        new syncurrent
                """
                if firing_delayed is None:
                    firing_delayed = firing.clone() 

                syncurrent = x @ self.weight_iv + firing_delayed @ self.weight_lat
                
                if self.ascs:
                        ascurrent = (ascurrent * self.transform_to_asc_r(self.trans_asc_r) + self.asc_amp) * firing * (self.dt/self.tau) + (1 - self.dt * self.transform_to_k(self.trans_asc_k)) * ascurrent
                
                voltage = syncurrent + self.dt * self.transform_to_k(self.trans_k_m) * self.R * (torch.sum(ascurrent, dim=0) + self.I0) + (1 - self.dt * self.transform_to_k(self.trans_k_m)) * voltage - (self.dt / self.tau) * firing * (voltage - self.v_reset)
                firing = self.spike_fn(voltage)
                return firing, voltage, ascurrent, syncurrent
        
        def transform_to_asc_r(self, param):
                """
                Transforms parameter to asc_r used in neuronal model
                """
                return 1 - (2 * torch.sigmoid(param)) # training on ln((1-r_j)/(1+r_j))
        
        def transform_to_k(self, param):
                """
                Transforms parameter to k used in neuronal model
                """
                return torch.sigmoid(param) / self.dt

class RNNC(nn.Module): 
        """
        Defines single recurrent layer ("recurrent neural network cell")

        Parameters
        ----------
        input_size : int
                number of dimensions in input
        hidden_size : int
                number of neurons
        bias : boolean, default False
                whether bias should be used
        """
        def __init__(self, input_size, hidden_size, bias = True):
                super().__init__()
                self.weight_ih = Parameter(torch.randn((input_size, hidden_size)))
                self.weight_lat = Parameter(torch.randn((hidden_size, hidden_size)))

                if bias:
                        self.bias = Parameter(torch.zeros((1, hidden_size)))
                else:
                        self.bias = torch.zeros((1, hidden_size))

                with torch.no_grad():
                    nn.init.uniform_(self.weight_ih, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))
                    nn.init.uniform_(self.weight_lat, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))
                    if bias:
                            nn.init.uniform_(self.bias, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))

        def forward(self, x, hidden, hidden_delayed, track = False):
                """
                Propagates single timestep
                
                Parameters
                ----------
                x : torch tensor (n, ndim)
                        input signal
                hidden : torch tensor (n, ndim)
                        previous hidden state
                track : bool, default False
                        whether to return pre-activation hidden state as well
                
                Return
                ------
                out : torch tensor (n, ndim)
                        updated hidden state
                hidden : torch tensor (n, ndim) [if track is True]
                        hidden state pre-activation
                """
                hidden = torch.mm(x, self.weight_ih) + torch.mm(hidden_delayed, self.weight_lat) + self.bias
                out = torch.tanh(hidden)
                if track:
                        return out, hidden
                return out
