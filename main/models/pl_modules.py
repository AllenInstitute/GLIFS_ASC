"""
This file describes three neural network models (GLIFRN, RNN, LSTMN),
along with a base module. Before utilizing any of these, be sure to
initialize with parameters specified by the class's add_model_specific_args
function and the add_structure_args add_general_model_args functions
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neurons import GLIFR, RNNC, NeuronParameters, StructureParameters
import pytorch_lightning as pl
from training.loss_fns import *
from utils.check import check_nonnegative_float, check_nonnegative_int, check_positive_float, check_positive_int

# TODO: Add required to parser add arguments
def add_structure_args(parent_parser):
    parser = parent_parser.add_argument_group("network_structure")
    parser.add_argument("--input_size", type=check_positive_int)
    parser.add_argument("--hidden_size", type=check_positive_int)
    parser.add_argument("--output_size", type=check_positive_int)
    parser.add_argument("--output_weight", type=bool) # Whether model should have final linear layer

def add_general_model_args(parent_parser):
    parser = parent_parser.add_argument_group("general_model_params")
    parser.add_argument("--dropout_prob", type=check_nonnegative_float, default=0)
    parser.add_argument("--synaptic_delay", type=check_nonnegative_float, default=0)
    parser.add_argument("--dt", type=check_positive_float, default=0.05)
    parser.add_argument("--final_reduction", default="none", type=str, choices=["none", "last"])

def add_training_args(parent_parser):
    parser = parent_parser.add_argument_group("train_params")
    parser.add_argument("--lr", type=check_positive_float, default=0.01)
#     parser.add_argument("--batch_size", type=check_positive_int, default=32)
    parser.add_argument("--loss_fn", type=str, default="F.mse_loss")
    parser.add_argument("--optimizer", type=str, default="torch.optim.Adam")
    parser.add_argument("--log_accuracy", type=bool, default=False)

class BaseModule(pl.LightningModule):
        """
        Base module for all networks desribed in file.
        """
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.save_hyperparameters()

            # Generate model
            self._init_model()

            # Compute number of timesteps for the synaptic delay
            self.delay = int(self.hparams.synaptic_delay / self.hparams.dt)
            if self.delay == 0:
                self.delay = 1

            # Initialize for silencing
            self.idx = []
            self.register_buffer("silence_mult", torch.eye(self.hparams.hidden_size))

            self.reset_state(full_reset=True)
        
        def forward(self, input, track=False):
                """
                Computes output of network given input.
                """
                return input
        
        def configure_optimizers(self):
                """
                Configures optimizer using one passed in on __init__.
                """
                optimizer = eval(self.hparams.optimizer)
                return optimizer(self.parameters(), lr=self.hparams.lr)
    
        def predict(self, batch, batch_idx):
                """
                Computes prediction on batch.
                """
                x, y = batch
                self.reset_state(len(y))
                return self(x, track=False)[:,-1,:] if self.hparams.final_reduction == 'last' else self(x, track=False)# TODO: Document

        def training_step(self, batch, batch_idx):
                return self._step(batch, batch_idx, "train")

        def validation_step(self, batch, batch_idx):
                self._step(batch, batch_idx, "val")

        def test_step(self, batch, batch_idx):
                self._step(batch, batch_idx, "test")

        def _step(self, batch, batch_idx, log_id):
                """
                Parameters:
                -----------
                - batch: Tuple
                        tuple of input, target where target is expected
                        to be one-hot encoding
                - batch_idx: int
                - log_id: str
                """
                loss_fn = eval(self.hparams.loss_fn)
                x, y = batch
                self.reset_state(len(y))
                y_hat = self(x, track=False)[:,-1,:] if self.hparams.final_reduction == 'last' else self(x, track=False)# TODO: Document
                loss = loss_fn(y_hat, y)

                if self.hparams.final_reduction == 'none':
                        # print(y.shape)
                        y_hat = y_hat[:, -1, :]

                if self.hparams.log_accuracy:
                        # print(y.shape)
                        _, y_hat_disc = torch.max(y_hat, 1)
                        _, y_disc = torch.max(y, 1)
                        self.log(f"{log_id}_acc", (y_hat_disc.cpu().detach().data == y_disc.cpu()).sum() / len(y), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

                self.log(f"{log_id}_loss", loss.cpu().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                return loss

        def _init_model(self):
                pass

        @staticmethod
        def add_model_specific_args(parent_parser):
                pass

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
                self.register_buffer("silence_mult", torch.eye(self.hparams.hidden_size))
                for i in self.idx:
                        self.silence_mult[i, i] = 0

class GLIFRN(BaseModule):
        """
        Fully connected GLIFR neural network.
        Defines a single recurrent layer network of GLIFR neurons with a synaptic delay.

        To use, __init__ requires arguments specified by add_structure_args,
        add_general_model_args, GLIFRN.add_model_specific_args
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def forward(self, input, track=False):
            """
            Propagates input through network.
            Parameters
            ----------
            input : Tensor(batch_size, nsteps, in_size)
                    input signal to be input over time
            track : bool, default False
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
            nsteps = input.shape[1]

            delay = self.delay

            # Initialize empty variables to track over time
            outputs = torch.empty((self.batch_size, nsteps, self.hparams.output_size)).type_as(input)
            if track:
                voltages = torch.empty((self.batch_size, nsteps, self.hparams.hidden_size)).type_as(input)
                ascs = torch.empty((self.hparams.num_ascs, self.batch_size, nsteps, self.hparams.hidden_size)).type_as(input)
                syns = torch.empty((self.batch_size, nsteps, self.hparams.hidden_size))
            outputs_ = [torch.zeros((self.batch_size, self.hparams.hidden_size)).type_as(input) if delay > 0 else None for i in range(delay + 1)]

            # Loop over input
            for step in range(nsteps):
                x = input[:, step, :]
                x = x.view(x.shape[0], -1)

                # Propagate through model
                self.firing, self.voltage, self.ascurrents, self.syncurrent = self.neuron_layer(x, self.firing, self.voltage, self.ascurrents, self.syncurrent, outputs_[-delay])
                self.firing = self.dropout_layer(self.firing)
                self.firing = torch.matmul(self.firing, self.silence_mult)
                outputs_.append(self.firing.clone() if delay > 0 else None)
                out = self.firing if self.output_linear is None else self.output_linear(self.firing)

                # Store outputs
                outputs[:, step, :] = out
                if track:
                    voltages[:, step, :] = self.voltage.clone()
                    ascs[:, :, step, :] = self.ascurrents.clone()
                    syns[:, step, :] = self.syncurrent.clone()
                
                if len(outputs_) > delay + 1:
                    outputs_ = outputs_[-delay-1:]
            # Return
            if track:
                    return outputs, voltages, ascs, syns 
            else:
                    return outputs
                
        def _init_model(self):
            """
            Initializes model with given parameters
            """
            self.neuron_layer = GLIFR(
                structure_params=StructureParameters(
                    input_size=self.hparams.input_size,
                    hidden_size=self.hparams.hidden_size,
                    output_size=self.hparams.output_size
                ),
                neuron_params = NeuronParameters(
                    dt=self.hparams.dt,
                    tau=self.hparams.tau,
                    num_ascs=self.hparams.num_ascs,
                    sigma_v=self.hparams.sigma_v,
                    R=self.hparams.R,
                    I0=self.hparams.I0,
                    v_reset=self.hparams.v_reset,
                    initialization=self.hparams.initialization
                )
            )
            self.output_linear = nn.Linear(
                    in_features = self.hparams.hidden_size, 
                    out_features = self.hparams.output_size, 
                    bias = True
                ) if self.hparams.output_weight else None
            self.dropout_layer = nn.Dropout(
                p=self.hparams.dropout_prob, 
                inplace=False
            )
            if not self.hparams.params_learned:
                    self.freeze_params_(self.dynamics_param_names_())

            # Initialize network
            if self.hparams.initialization == 'het':
                ckpt_model = GLIFRN.load_from_checkpoint(self.hparams.ckpt_path, strict=False)
                print(f"Initializing from {self.hparams.ckpt_path}")
                # self.neuron_layer.het_init_()
                with torch.no_grad():
                        thresh = ckpt_model.neuron_layer.thresh.detach().cpu().numpy().reshape(-1)
                        thresh_shuffle = np.random.choice(thresh, len(self.neuron_layer.thresh.reshape(-1)))
                        self.neuron_layer.thresh.copy_(torch.from_numpy(thresh_shuffle).reshape(self.neuron_layer.thresh.shape))

                        
                        trans_k_m = ckpt_model.neuron_layer.trans_k_m.detach().cpu().numpy().reshape(-1)
                        trans_k_m_shuffle = np.random.choice(trans_k_m, len(self.neuron_layer.trans_k_m.reshape(-1)))
                        self.neuron_layer.trans_k_m.copy_(torch.from_numpy(trans_k_m_shuffle).reshape(self.neuron_layer.trans_k_m.shape))

                        if self.hparams.num_ascs > 0:
                                trans_k_j = ckpt_model.neuron_layer.trans_k_j.detach().cpu().numpy().reshape(-1)
                                trans_k_j_shuffle = np.random.choice(trans_k_j, len(self.neuron_layer.trans_k_j.reshape(-1)))
                                self.neuron_layer.trans_k_j.copy_(torch.from_numpy(trans_k_j_shuffle).reshape(self.neuron_layer.trans_k_j.shape))

                                
                                a_j = ckpt_model.neuron_layer.a_j.detach().cpu().numpy().reshape(-1)
                                a_j_shuffle = np.random.choice(a_j, len(self.neuron_layer.a_j.reshape(-1)))
                                self.neuron_layer.a_j.copy_(torch.from_numpy(a_j_shuffle).reshape(self.neuron_layer.a_j.shape))

                                
                                trans_r_j = ckpt_model.neuron_layer.trans_r_j.detach().cpu().numpy().reshape(-1)
                                trans_r_j_shuffle = np.random.choice(trans_r_j, len(self.neuron_layer.trans_r_j.reshape(-1)))
                                self.neuron_layer.trans_r_j.copy_(torch.from_numpy(trans_r_j_shuffle).reshape(self.neuron_layer.trans_r_j.shape))
                        
                        weight_iv = ckpt_model.neuron_layer.weight_iv.detach().cpu().numpy().reshape(-1)
                        weight_iv_shuffle = np.random.choice(weight_iv, len(self.neuron_layer.weight_iv.reshape(-1)))
                        self.neuron_layer.weight_iv.copy_(torch.from_numpy(weight_iv_shuffle).reshape(self.neuron_layer.weight_iv.shape))

                        weight_lat = ckpt_model.neuron_layer.weight_lat.detach().cpu().numpy().reshape(-1)
                        weight_lat_shuffle = np.random.choice(weight_lat, len(self.neuron_layer.weight_lat.reshape(-1)))
                        self.neuron_layer.weight_lat.copy_(torch.from_numpy(weight_lat_shuffle).reshape(self.neuron_layer.weight_lat.shape))

            elif self.hparams.initialization == 'hom':
                self.neuron_layer.hom_init_()
            else:
                raise ValueError("initialization must be either het or hom")
            firing, voltage, syncurrent, ascurrents = self.neuron_layer.init_states(1)

            self.register_buffer("firing", firing, persistent=False)
            self.register_buffer("voltage", voltage, persistent=False)
            self.register_buffer("syncurrent", syncurrent, persistent=False)
            self.register_buffer("ascurrents", ascurrents, persistent=False)

        @staticmethod
        def add_model_specific_args(parent_parser):
            BaseModule.add_model_specific_args(parent_parser)

            parser = parent_parser.add_argument_group("glifr")
            parser.add_argument("--tau", type=check_positive_float, default=0.05)
            parser.add_argument("--num_ascs", type=check_nonnegative_int, default=2)
            parser.add_argument("--sigma_v", type=check_positive_float, default=1)
            parser.add_argument("--R", type=check_positive_float, default=0.1)
            parser.add_argument("--I0", type=float, default=0)
            parser.add_argument("--v_reset", type=float, default=0)
            parser.add_argument("--initialization", type=str, choices=['het', 'hom'], default="hom") 
            parser.add_argument("--params_learned", type=bool, default=True)     
            parser.add_argument("--ckpt_path", type=str, default="")        

        def freeze_params_(self, param_names):
            """
            Freezes parameters, preventing training
            """
            self.neuron_layer.freeze_params_(param_names)

        def membrane_param_names_(self):
                """
                Returns a list of names of membrane related parameters
                """
                return self.neuron_layer.membrane_param_names_()

        def asc_param_names_(self):
                """
                Returns a list of names of after-spike current related parameters
                """
                return self.neuron_layer.asc_param_names_()
        
        def dynamics_param_names_(self):
                """
                Returns a list of names of both membrane-related and
                after-spike current related parameters
                """
                return self.neuron_layer.dynamics_param_names_()
        
        def learnable_params_(self):
                """
                Returns a list of names of all learnable parameters
                """
                learnable = []
                for name, param in self.named_parameters():
                        if param.requires_grad:
                                learnable.append(name)
                return learnable

        def reset_state(self, batch_size = 1, full_reset = True):
                """
                Resets internal state of network
                Parameters
                ----------
                batch_size : int, default 1
                        batch size to be used subsequently
                full_reset : boolean, default True
                        whether internal states should be re-initialized
                """
                self.batch_size = batch_size
                
                if full_reset:
                    firing, voltage, syncurrent, ascurrents = self.neuron_layer.init_states(self.batch_size)
                    self.firing = firing.type_as(self.firing)
                    self.voltage = firing.type_as(self.voltage)
                    self.syncurrent = firing.type_as(self.syncurrent)
                    self.ascurrents = firing.type_as(self.ascurrents)
                else:
                    self.firing.detach_()
                    self.voltage.detach_()
                    self.syncurrent.detach_()
                    self.ascurrents.detach_()

class RNN(BaseModule):
        """
        Fully connected biological neural network.
        Defines a single recurrent layer network of GLIFR neurons with a synaptic delay.
        Parameters
        ----------
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def forward(self, input, track=False):
                """
                Propagates input through network.

                Parameters
                ----------
                input : Tensor(batch_size, nsteps, in_size)
                        input signal to be input over time
                track : bool, default False
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

                nsteps = input.shape[1]

                # Set up tensors to store results
                outputs = torch.empty((self.batch_size, nsteps, self.hparams.output_size)).type_as(input)
                outputs_ = [torch.zeros((self.batch_size, self.hparams.hidden_size)).type_as(input) for i in range(delay)]

                if track:
                        voltages = torch.empty((self.batch_size, nsteps, self.hparams.hidden_size)).type_as(input)
                
                # Propagate through network
                for step in range(nsteps):
                        # Get input
                        x = input[:, step, :]
                        x = x.view(x.shape[0], -1)
                        self.firing, voltage = self.neuron_layer(x, self.firing, outputs_[-delay], track=True)
                        self.firing = self.dropout_layer(self.firing)
                        self.firing = torch.matmul(self.firing, self.silence_mult)
                        out = self.firing if self.output_linear is None else self.output_linear(self.firing)
                        outputs[:, step, :] = out

                        outputs_.append(self.firing.clone())
                        if len(outputs_) > delay:
                            outputs_ = outputs_[-delay:]
                        if track:
                                voltages[:, step, :] = voltage

                if track:
                        return outputs, voltages
                return outputs
        
        def _init_model(self):
                self.output_linear = nn.Linear(
                        in_features = self.hparams.hidden_size, 
                        out_features = self.hparams.output_size, 
                        bias = True
                ) if self.hparams.output_weight else None
                self.neuron_layer = RNNC(StructureParameters(
                        input_size = self.hparams.input_size,
                        hidden_size = self.hparams.hidden_size,
                        output_size = self.hparams.output_size
                ), bias = True)
                self.dropout_layer = nn.Dropout(
                        p=self.hparams.dropout_prob, 
                        inplace=False
                )
                # self.firing = torch.zeros((1, self.hparams.hidden_size)).to(self.device)
                self.register_buffer("firing", torch.zeros((1, self.hparams.hidden_size)), persistent=False)


        @staticmethod
        def add_model_specific_args(parent_parser):
                BaseModule.add_model_specific_args(parent_parser)

        def reset_state(self, batch_size = 1, full_reset = True):
                """
                Resets internal state of network

                Parameters
                ----------
                batch_size : int, default 1
                        batch size to be used subsequently
                """
                self.batch_size = batch_size

                if full_reset:
                        self.firing = torch.zeros((self.batch_size, self.hparams.hidden_size)).type_as(self.firing)
                else:
                        self.firing.detach_()

class LSTMN(BaseModule):
        """
        Fully connected biological neural network.
        Defines a single recurrent layer network of GLIFR neurons with a synaptic delay.
        Parameters
        ----------
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def forward(self, input, track=False):
                """
                Propagates input through network.

                Parameters
                ----------
                input : Tensor(batch_size, nsteps, in_size)
                        input signal to be input over time
                track : bool, default False
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

                nsteps = input.shape[1]

                # Set up tensors to store results
                outputs = torch.empty((self.batch_size, nsteps, self.hparams.output_size)).type_as(input)
                outputs_ = [torch.zeros((self.batch_size, self.hparams.hidden_size)).type_as(input) for i in range(delay)]

                if track:
                        voltages = torch.empty((self.batch_size, nsteps, self.hparams.hidden_size)).type_as(input)
                
                # Propagate through network
                for step in range(nsteps):
                        # Get input
                        x = input[:, step, :]
                        x = x.view(x.shape[0], -1)
                        self.firing, self.c = self.neuron_layer(x, (self.firing, self.c))
                        self.firing = self.dropout_layer(self.firing)
                        self.firing = torch.matmul(self.firing, self.silence_mult)
                        out = self.firing if self.output_linear is None else self.output_linear(self.firing)
                        outputs[:, step, :] = out

                        outputs_.append(self.firing.clone())
                        if len(outputs_) > delay:
                            outputs_ = outputs_[-delay:]
                        if track:
                                voltages[:, step, :] = voltage

                if track:
                        return outputs, voltages
                return outputs
        
        def _init_model(self):
                self.output_linear = nn.Linear(
                        in_features = self.hparams.hidden_size, 
                        out_features = self.hparams.output_size, 
                        bias = True
                ) if self.hparams.output_weight else None
                self.neuron_layer = nn.LSTMCell(
                        input_size = self.hparams.input_size, 
                        hidden_size = self.hparams.hidden_size, 
                        bias = True
                )
                self.dropout_layer = nn.Dropout(
                        p=self.hparams.dropout_prob, 
                        inplace=False
                )
                # self.firing = torch.zeros((1, self.hparams.hidden_size)).to(self.device)
                self.register_buffer("firing", torch.zeros((1, self.hparams.hidden_size)), persistent=False)

        @staticmethod
        def add_model_specific_args(parent_parser):
                BaseModule.add_model_specific_args(parent_parser)

        def reset_state(self, batch_size = 1, full_reset = True):
                """
                Resets internal state of network

                Parameters
                ----------
                batch_size : int, default 1
                        batch size to be used subsequently
                """
                self.batch_size = batch_size

                if full_reset:
                        self.firing = torch.zeros((self.batch_size, self.hparams.hidden_size)).type_as(self.firing)
                        self.c = torch.zeros((self.batch_size, self.hparams.hidden_size)).type_as(self.firing)
                else:
                        self.firing.detach_()
