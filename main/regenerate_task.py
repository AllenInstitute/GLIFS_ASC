"""
This file trains a single rate-based GLIF neuron with after-spike currents to produce realizable outputs.

Initial output of neuron saved to
f"results/{base_name_results}-initialoutputs.csv"

Final output of neuron saved to
f"results/{base_name_results}-finaloutputs.csv"

Target output for neuron saved to
f"results/{base_name_results}-targets.csv"

Final membrane parameters are saved to
f"results/{base_name_results}-membraneparams.csv"
where the first column lists thresh and the second column lists k_m

Final after-spike current related parameters are saved to
f"results/{base_name_results}-ascparams.csv"
where the first column lists k_j, the second column lists r_j, and the third column lists a_j.

Values of * over training are saved to
f"results/{base_name_results}-{hid_size}units-{*}overlearning.csv"
where * can be any of ["km", "thresh", "asck", "ascr", "ascamp"]

MSE losses over training are saved to
f"results/{base_name_results}-{hid_size}units-losses.csv"

PyTorch model dictionary for trained learning model saved to
f"saved_models/{base_name_model}_learned.pt"

PyTorch model dictionary for target model saved to
f"saved_models/{base_name_model}_target.pt"

Dictionary with information over training is saved to
f"traininfo/{base_name_save}.pickle"
"""

import pickle
from models.networks import BNNFC
from pylab import cm


import numpy as np

import torch
import torch.nn as nn
import math

def main():
    main_name = "test"

    base_name_save = "test/" + main_name
    base_name_model = "test/" + main_name
    base_name_results = "test/" + main_name

    training_info = {"losses": [],
        "weights": [],
        "threshes": [],
        "v_resets": [],
        "k_ms": [],
        "k_syns": [],
        "asc_amps": [],
        "asc_rs": [],
        "asc_ks": []
        }

    dt = 0.05
    sim_time = 10
    nsteps = int(sim_time / dt)

    hid_size = 1
    input_size = 1
    output_size = 1

    targets = torch.empty((1, nsteps, output_size))
    inputs = 0.01 * torch.ones((1, nsteps, input_size))

    train_params = ["thresh", "k_m", "asc_amp", "asc_r", "asc_k"]#, "k_m"]

    target_model = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, dt=dt, output_weight=False)
    learning_model = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, dt=dt, output_weight=False)

    # Adjust target model
    target_model.neuron_layer.weight_iv.data = (1 / hid_size) * torch.ones((target_model.neuron_layer.input_size, target_model.neuron_layer.hidden_size))

    with torch.no_grad():
        asc_r1, asc_r2 = 1 - 1e-10, -(1 - 1e-10)
        asc_amp1, asc_amp2 = -1, 1
        target_model.neuron_layer.trans_asc_r[0,:,:] = math.log((1 - asc_r1) / (1 + asc_r1))
        target_model.neuron_layer.trans_asc_r[1,:,:] = math.log((1 - asc_r2) / (1 + asc_r2))

        target_model.neuron_layer.asc_amp[0,:,:] = asc_amp1
        target_model.neuron_layer.asc_amp[1,:,:] = asc_amp2

    # Equate learning model to target model
    with torch.no_grad():
        learning_model.neuron_layer.thresh.data = target_model.neuron_layer.thresh.data
        learning_model.neuron_layer.trans_k_m.data = target_model.neuron_layer.trans_k_m.data
        learning_model.neuron_layer.asc_amp.data = target_model.neuron_layer.asc_amp.data
        learning_model.neuron_layer.trans_asc_r.data = target_model.neuron_layer.trans_asc_r.data
        learning_model.neuron_layer.trans_asc_k.data = target_model.neuron_layer.trans_asc_k.data
        learning_model.neuron_layer.weight_iv.data = target_model.neuron_layer.weight_iv.data

        # Modify learning model
        if "asc_amp" in train_params:
            learning_model.neuron_layer.asc_amp.data = 0 * torch.randn((2, 1, hid_size), dtype=torch.float)
        if "asc_r" in train_params:
            learning_model.neuron_layer.trans_asc_r.data = torch.randn((2, 1, hid_size), dtype=torch.float)
        if "asc_k" in train_params:
            learning_model.neuron_layer.trans_asc_k.data = torch.randn((2, 1, hid_size), dtype=torch.float)
        if "k_m" in train_params:
            new_km = 1
            learning_model.neuron_layer.trans_k_m.data = math.log(new_km * dt / (1 - (new_km * dt))) * torch.ones((1, hid_size), dtype=torch.float)
        if "thresh" in train_params:
            learning_model.neuron_layer.thresh.data = torch.randn((1, hid_size), dtype=torch.float)

    target_model.eval()
    # Compute target output
    with torch.no_grad():
        target_model.reset_state(1)
        outputs = target_model(inputs)
        targets[0,:,:] = outputs[0, -nsteps:, :]
    
    # Collect initial response of learning neuron
    with torch.no_grad():
        learning_model.reset_state(1)
        outputs = learning_model(inputs)
        outputs[0,:,:] = outputs[0, -nsteps:, :]
        np.savetxt("results/" + base_name_results + "-" + "initialoutputs.csv", np.stack(outputs).reshape((-1, 1)), delimiter=',')

    # Train model
    num_epochs = 6000
    lr = 0.001#0.003# 0.01 for thresh, 0.1 for asck

    train_params_real = []
    if "thresh" in train_params:
        train_params_real.append(learning_model.neuron_layer.thresh)
    if "k_m" in train_params:
        train_params_real.append(learning_model.neuron_layer.trans_k_m)
    if "asc_amp" in train_params:
        train_params_real.append(learning_model.neuron_layer.asc_amp)
    if "asc_k" in train_params:
        train_params_real.append(learning_model.neuron_layer.trans_asc_k)
    if "asc_r" in train_params:
        train_params_real.append(learning_model.neuron_layer.trans_asc_r)

    optimizer = torch.optim.Adam(train_params_real, lr=lr)
    losses = []
    loss_fn = nn.MSELoss()
    learning_model.train()
    for epoch in range(num_epochs):
        loss = 0.0
        learning_model.reset_state(1)
        optimizer.zero_grad()

        outputs = learning_model(inputs)
        loss = loss + loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch}/{num_epochs}: loss of {loss.item()}")
        losses.append(loss.item())

        with torch.no_grad():
            print(learning_model.neuron_layer.thresh.grad)
            training_info["k_ms"].append([learning_model.neuron_layer.transform_to_k(learning_model.neuron_layer.trans_k_m[0,j]).item() - learning_model.neuron_layer.transform_to_k(target_model.neuron_layer.trans_k_m[0,j]).item()  + 0.0 for j in range(learning_model.hid_size)])
            training_info["threshes"].append([learning_model.neuron_layer.thresh[0,j].item() - target_model.neuron_layer.thresh[0,j].item()  + 0.0 for j in range(learning_model.hid_size)])
            training_info["asc_ks"].append([learning_model.neuron_layer.transform_to_k(learning_model.neuron_layer.trans_asc_k[j,0,m]).item() - learning_model.neuron_layer.transform_to_k(target_model.neuron_layer.trans_asc_k[j,0,m]).item()  + 0.0 for j in range(learning_model.neuron_layer.num_ascs) for m in range(learning_model.hid_size)])
            training_info["asc_amps"].append([learning_model.neuron_layer.asc_amp[j,0,m].item() - target_model.neuron_layer.asc_amp[j,0,m].item()  + 0.0 for j in range(learning_model.neuron_layer.num_ascs) for m in range(learning_model.hid_size)])
            training_info["asc_rs"].append([learning_model.neuron_layer.transform_to_asc_r(learning_model.neuron_layer.trans_asc_r)[j,0,m].item() - learning_model.neuron_layer.transform_to_asc_r(target_model.neuron_layer.trans_asc_r)[j,0,m].item()  + 0.0 for j in range(learning_model.neuron_layer.num_ascs) for m in range(learning_model.hid_size)])
            training_info["losses"].append(loss.item())

    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "kmoverlearning.csv", np.array(training_info["k_ms"]), delimiter=",")
    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "threshoverlearning.csv", np.array(training_info["threshes"]), delimiter=",")
    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "asckoverlearning.csv", np.array(training_info["asc_ks"]), delimiter=",")
    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "ascroverlearning.csv", np.array(training_info["asc_rs"]), delimiter=",")
    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "ascampoverlearning.csv", np.array(training_info["asc_amps"]), delimiter=",")

    with torch.no_grad():
        learning_model.reset_state(1)
        outputs = learning_model(inputs)

        outputs[0,:,:] = outputs[0, -nsteps:, :]

        np.savetxt("results/" + base_name_results + "-" + "finaloutputs.csv", np.stack(outputs).reshape((-1, 1)), delimiter=',')
        np.savetxt("results/" + base_name_results + "-" + "targets.csv", targets.reshape((-1,1)), delimiter=',')

    np.savetxt("results/" + base_name_results + "-" + str(hid_size) + "units-" + "losses.csv", np.array(losses), delimiter=",")

    membrane_parameters = np.zeros((hid_size, 2))
    membrane_parameters[:, 0] = learning_model.neuron_layer.thresh.detach().numpy().reshape(-1) - target_model.neuron_layer.thresh.detach().numpy().reshape(-1)
    membrane_parameters[:, 1] = learning_model.neuron_layer.transform_to_k(learning_model.neuron_layer.trans_k_m).detach().numpy().reshape(-1) - target_model.neuron_layer.transform_to_k(target_model.neuron_layer.trans_k_m).detach().numpy().reshape(-1)
    np.savetxt("results/" + base_name_results + "-" + "membraneparams.csv", membrane_parameters, delimiter=',')

    asc_parameters = np.zeros((hid_size * 2, 3))
    asc_parameters[:, 0] = learning_model.neuron_layer.transform_to_k(learning_model.neuron_layer.trans_asc_k)[:,0,:].detach().numpy().reshape(-1) - target_model.neuron_layer.transform_to_k(target_model.neuron_layer.trans_asc_k)[:,0,:].detach().numpy().reshape(-1)
    asc_parameters[:, 1] = learning_model.neuron_layer.transform_to_asc_r(learning_model.neuron_layer.trans_asc_r)[:,0,:].detach().numpy().reshape(-1) - target_model.neuron_layer.transform_to_asc_r(target_model.neuron_layer.trans_asc_r)[:,0,:].detach().numpy().reshape(-1)
    asc_parameters[:, 2] = learning_model.neuron_layer.asc_amp[:,0,:].detach().numpy().reshape(-1) - target_model.neuron_layer.asc_amp[:,0,:].detach().numpy().reshape(-1)
    np.savetxt("results/" + base_name_results + "-ascparams.csv", asc_parameters, delimiter=',')

    torch.save(learning_model.state_dict(), "saved_models/" + base_name_model + "_learned.pt")
    torch.save(target_model.state_dict(), "saved_models/" + base_name_model + "_target.pt")
    torch.save(training_info["losses"], "traininfo/" + base_name_save + "_losses.pt")

    with open("traininfo/" + base_name_save + ".pickle", 'wb') as handle:
                pickle.dump(training_info, handle)

if __name__ == '__main__':
        main()
