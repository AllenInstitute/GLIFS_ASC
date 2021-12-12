"""
This file is used to save the parameters of models to a file.
In order to use it, replace base_name_results and base_name_model
to point to the beginning of the output results file and the input saved
model name respectively.

Parameters of model will be saved to 
f""results/{base_name_results}-{hid_size}units-0itr-init-allparams.csv" if init is True
f""results/{base_name_results}-{hid_size}units-0itr-allparams.csv"
Parameters will be taken from model
f""saved_models/{base_name_model}-{hid_size}units-0itr-init.pt" if init is True
f""saved_models/{base_name_model}-{hid_size}units-0itr.pt"
"""

import numpy as np
import torch

from models.networks import BNNFC

fontsize = 18
main_name = "smnist-4-final"
base_name_results = "results/paper_results/smnist_results/" + main_name
base_name_model = "models_wkof_080821/" + main_name

init = False
input_size = 28 
hid_size = 256
output_size = 10

model_glif = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size)

if init:
    model_glif.load_state_dict(torch.load("saved_models/" + base_name_model + "-" + str(hid_size) + "units-" + str(0) + "itr-init.pt"))
else:
    model_glif.load_state_dict(torch.load("saved_models/" + base_name_model + "-" + str(hid_size) + "units-" + str(0) + "itr.pt"))

parameters = np.zeros((hid_size, 8))
parameters[:, 0] = model_glif.neuron_layer.thresh.detach().numpy().reshape(-1)
parameters[:, 1] = model_glif.neuron_layer.transform_to_k(model_glif.neuron_layer.trans_k_m).detach().numpy().reshape(-1)
parameters[:, 2] = model_glif.neuron_layer.transform_to_asc_r(model_glif.neuron_layer.trans_asc_r).detach().numpy()[0,0,:]
parameters[:, 3] = model_glif.neuron_layer.transform_to_asc_r(model_glif.neuron_layer.trans_asc_r).detach().numpy()[1,0,:]
parameters[:, 4] = model_glif.neuron_layer.transform_to_k(model_glif.neuron_layer.trans_asc_k).detach().numpy()[0,0,:]
parameters[:, 5] = model_glif.neuron_layer.transform_to_k(model_glif.neuron_layer.trans_asc_k).detach().numpy()[1,0,:]
parameters[:, 6] = model_glif.neuron_layer.asc_amp.detach().numpy()[0,0,:]
parameters[:, 7] = model_glif.neuron_layer.asc_amp.detach().numpy()[1,0,:]

addinit = "-init" if init else ""
np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + str(0) + "itr" + addinit + "-allparams.csv", parameters, delimiter=',')
