"""
This file defines models for single layers of neurons.
"""
import logging
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from utils.types import NeuronParameters, StructureParameters

class GLIFR(nn.Module):
        """
        Defines a single layer of GLIFR-ASC neurons
        Parameters
        ----------
        """
        def __init__(self, structure_params: StructureParameters, neuron_params: NeuronParameters):
                super().__init__()
                self.input_size = structure_params.input_size
                self.hidden_size = structure_params.hidden_size
                self.num_ascs = neuron_params.num_ascs

                self.dt = neuron_params.dt
                self.tau = neuron_params.tau
                if self.tau is None:
                        self.tau = self.dt
                self.sigma_v = neuron_params.sigma_v

                self.weight_iv = Parameter(torch.randn((self.input_size, self.hidden_size))) # incoming weights
                self.weight_lat = Parameter(torch.randn((self.hidden_size, self.hidden_size))) # lateral connections (i.e., among neurons in same layer              
                self.thresh = Parameter(torch.randn((1, self.hidden_size)))
                self.trans_k_m = Parameter(torch.randn((1, self.hidden_size)))
                if self.num_ascs > 0:
                        self.trans_k_j = Parameter(torch.randn((self.num_ascs, 1, self.hidden_size)))
                        self.trans_r_j = Parameter(torch.randn((self.num_ascs, 1, self.hidden_size)))
                        self.a_j = Parameter(torch.randn((self.num_ascs, 1, self.hidden_size)))
                self.v_reset = neuron_params.v_reset

                self.R = neuron_params.R
                self.I0 = neuron_params.I0
                
                # Freeze parameters
                if neuron_params.num_ascs == 0:
                        self.freeze_params_(self.asc_param_names_())

        # def het_init_(self):
                # with torch.no_grad():
                #         nn.init.uniform_(self.thresh, -1, 1)
                #         nn.init.zeros_(self.trans_k_m)
                #         k_m = 0.04 + 0.02 * torch.rand((1, self.hidden_size), dtype=torch.float)
                #         self.trans_k_m += torch.log((k_m * self.dt) / (1 - (k_m * self.dt)))
                        
                #         if self.num_ascs > 0:
                #                 nn.init.zeros_(self.trans_k_j)
                #                 k_j = 1.5 + torch.rand((self.num_ascs, 1, self.hidden_size), dtype=torch.float)
                #                 self.trans_k_j += torch.log((k_j * self.dt) / (1 - (k_j * self.dt)))

                #                 nn.init.zeros_(self.trans_r_j)
                #                 r_j = -0.1 + 0.2 * torch.rand((self.num_ascs, 1, self.hidden_size), dtype=torch.float)
                #                 self.trans_r_j += torch.log((1 - r_j) / (1 + r_j))

                #                 nn.init.uniform_(self.a_j, -0.1, 0.1)

                #         nn.init.uniform_(self.weight_iv, -math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))
                #         nn.init.uniform_(self.weight_lat, -math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))
        
        def hom_init_(self):
                with torch.no_grad():
                        nn.init.ones_(self.thresh)
                        nn.init.constant_(self.trans_k_m, math.log(self.dt * 0.05 / (1 - (0.05 * 0.05))))
                       
                        if self.num_ascs > 0:
                                nn.init.constant_(self.trans_k_j, math.log((0.1 / self.dt) * self.dt / (1 - ((0.1 / self.dt) * self.dt))))
                        
                                nn.init.zeros_(self.trans_r_j)
                                r_j = -0.01 + 0.02 * torch.rand((self.num_ascs, 1, self.hidden_size), dtype=torch.float)
                                self.trans_r_j += torch.log((1 - r_j) / (1 + r_j))

                                nn.init.uniform_(self.a_j, -0.01, 0.01)

                        nn.init.uniform_(self.weight_iv, -math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))
                        nn.init.uniform_(self.weight_lat, -math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))
        
        def freeze_params_(self, param_names):
                found = []
                for name, param in self.named_parameters():
                        if name in param_names:
                                found.append(name)
                                param.requires_grad = False
                for name in param_names:
                        if name not in found:
                                logging.error(f"{name} was not frozen")

        def learnable_params_(self):
                found = []
                for name, param in self.named_parameters():
                        if param.requires_grad:
                                found.append(name)
                return found

        def membrane_param_names_(self):
                return ["thresh", "trans_k_m"]

        def asc_param_names_(self):
                return ["trans_k_j", "trans_r_j", "a_j"]

        def dynamics_param_names_(self):
                return ["thresh", "trans_k_m", "trans_k_j", "trans_r_j", "a_j"]

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
                activation = (x - self.thresh) / self.sigma_v
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

                syncurrent = x @ self.weight_iv 
                if firing_delayed is not None:
                        syncurrent = syncurrent + firing_delayed @ self.weight_lat
                
                # TODO: Incorporate without asc
                if self.num_ascs > 0:
                        ascurrent = (ascurrent * self.transform_to_asc_r(self.trans_r_j) + self.a_j) * firing * (self.dt/self.tau) + (1 - self.dt * self.transform_to_k(self.trans_k_j)) * ascurrent
                else:
                        ascurrent = ascurrent * 0

                voltage = syncurrent + self.dt * self.transform_to_k(self.trans_k_m) * self.R * (torch.sum(ascurrent, dim=0) + self.I0) + (1 - self.dt * self.transform_to_k(self.trans_k_m)) * voltage - (self.dt / self.tau) * firing * (voltage - self.v_reset)
                firing = self.spike_fn(voltage)
                return firing, voltage, ascurrent, syncurrent
        
        def init_states(self, batch_size):
                firing = torch.zeros((batch_size, self.hidden_size))
                voltage = torch.zeros((batch_size, self.hidden_size))
                syncurrent = torch.zeros((batch_size, self.hidden_size))
                ascurrents = torch.zeros((self.num_ascs, batch_size, self.hidden_size))
                return firing, voltage, syncurrent, ascurrents
                
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


# class ConvGLIFR(GLIFR):
#         def __init__(self, structure_params: StructureParameters, neuron_params: NeuronParameters):
#                 super().__init__(structure_params=structure_params, neuron_params=neuron_params)


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
        def __init__(self, structure_params: StructureParameters, bias = True):
                super().__init__()
                self.input_size = structure_params.input_size
                self.hidden_size = structure_params.hidden_size

                self.weight_ih = Parameter(torch.randn((self.input_size, self.hidden_size)))
                self.weight_lat = Parameter(torch.randn((self.hidden_size, self.hidden_size)))

                if bias:
                        self.bias = Parameter(torch.zeros((1, self.hidden_size)))
                else:
                        self.bias = torch.zeros((1, self.hidden_size))

                with torch.no_grad():
                    nn.init.uniform_(self.weight_ih, -math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))
                    nn.init.uniform_(self.weight_lat, -math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))
                    if bias:
                            nn.init.uniform_(self.bias, -math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))

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
                if hidden_delayed is not None:
                        hidden = torch.mm(x, self.weight_ih) + torch.mm(hidden_delayed, self.weight_lat) + self.bias
                else:
                        hidden = torch.mm(x, self.weight_ih) + torch.mm(hidden, self.weight_lat) + self.bias
                out = torch.tanh(hidden)
                if track:
                        return out, hidden
                return out
        def learnable_params_(self):
            return [] #TODO
