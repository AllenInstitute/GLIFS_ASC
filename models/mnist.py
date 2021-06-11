import matplotlib
matplotlib.use('Agg')

# author: @chloewin
# 03/07/21
import pickle
import datetime
import utils as ut
from networks import RBNN, RNNFC, BNNFC
from neurons.glif_new import BNNC, RNNC, Placeholder

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as tud
import torch.nn as nn
import math
# import torch.utils.data.DataLoader

# torch.autograd.set_detect_anomaly(True)
"""
This file trains a network of rate-based GLIF neurons with after-spike currents on a pattern generation task.
1. Single pattern generation: generate a sinusoid of a given frequency when provided with constant input
2. Sussillo pattern generation: generate a sinusoid of a freuqency that is proportional to the amplitude of the constant input
3. Bellec pattern generation: generation a sinusoid of a frequency that corresponds to the subset of input neurons receiving input

Trained model is saved to the folder specified by model_name + date.
Figures on learned outputs, parameters, weights, gradients, and losses over training are saved to the folder specified by fig_name + date

Loss is printed on every epoch

To alter model architecture, change sizes, layers, and conns dictionaries. 
There are other specifications including amount of time, number of epochs, learning rate, etc.
"""

def main():
        # sim_time = 4
        # dt = 0.05
        # nsteps = int(sim_time / dt)

        # neuron = BNNC(input_size = 1, hidden_size = 1)
        # nn.init.constant_(neuron.ln_k_m, math.log(0.1))
        # nn.init.uniform_(neuron.thresh, -10, 10)
        # # nn.init.uniform_(neuron.asc_amp, -2, 2)
        # # nn.init.uniform_(neuron.asc_r, -2, 2)
        # # nn.init.uniform_(neuron.ln_asc_k, -3, 3)

        # initials = torch.empty((1, nsteps, 1))
        # input = torch.ones((1, nsteps, 1))
        # firing = torch.zeros((1, 1))
        # voltage = torch.zeros((1, 1))
        # syncurrent = torch.zeros((1, 1))
        # ascurrents = torch.zeros((2, 1, 1))

        # for step in range(nsteps):
        #       x = input[:, step, :]
        #       firing, voltage, ascurrents, syncurrent = neuron(x, firing, voltage, ascurrents, syncurrent)
        #       initials[:, step, :] = firing.detach()

        # neuron_init = BNNC(input_size = 1, hidden_size = 1)
        # targets = torch.empty((1, nsteps, 1))
        # input = torch.ones((1, nsteps, 1))
        # firing = torch.zeros((1, 1))
        # voltage = torch.zeros((1, 1))
        # syncurrent = torch.zeros((1, 1))
        # ascurrents = torch.zeros((2, 1, 1))

        # for step in range(nsteps):
        #       x = input[:, step, :]
        #       firing, voltage, ascurrents, syncurrent = neuron_init(x, firing, voltage, ascurrents, syncurrent)
        #       targets[:, step, :] = firing.detach()

        # optimizer = torch.optim.Adam(neuron.parameters(), lr = 0.025)
        # loss_fn = nn.MSELoss()
        # losses = []
        # k_ms = []
        # threshes = []
        # asc_rs = []
        # asc_ks = []
        # asc_amps = []
        # weights = []
        # for i in range(120):
        #       if i % 10 == 0:
        #               print(i)
        #       loss = 0.0
        #       optimizer.zero_grad()
        #       outputs = torch.empty((1, nsteps, 1))
        #       input = torch.ones((1, nsteps, 1))
        #       firing = torch.zeros((1, 1))
        #       voltage = torch.zeros((1, 1))
        #       syncurrent = torch.zeros((1, 1))
        #       ascurrents = torch.zeros((2, 1, 1))

        #       for step in range(nsteps):
        #               x = input[:, step, :]
        #               firing, voltage, ascurrents, syncurrent = neuron(x, firing, voltage, ascurrents, syncurrent)
        #               outputs[:, step, :] = firing


        #       k_ms.append([torch.exp(neuron.ln_k_m[0,j]) - torch.exp(neuron_init.ln_k_m[0,j])  + 0.0 for j in range(1)])
        #       threshes.append([neuron.thresh[0,j] - neuron_init.thresh[0,j]  + 0.0 for j in range(1)])
        #       asc_ks.append([torch.exp(neuron.ln_asc_k[j,0,m]) - torch.exp(neuron_init.ln_asc_k[j,0,m])  + 0.0 for j in range(2) for m in range(1)])
        #       asc_amps.append([neuron.asc_amp[j,0,m] - neuron_init.asc_amp[j,0,m]  + 0.0 for j in range(2) for m in range(1)])
        #       asc_rs.append([neuron.asc_r[j,0,m] - neuron_init.asc_r[j,0,m] + 0.0 for j in range(2) for m in range(1)])
        #       weights.append([torch.mean(neuron.weight_iv[:,m] - neuron_init.weight_iv[:,m])  + 0.0 for m in range(1)])

        #       loss = loss + loss_fn(outputs, targets)
        #       loss.backward()
        #       optimizer.step()
        #       losses.append(loss.item())

        # fontsize = 16
        # plt.plot(losses)
        # plt.xticks(fontsize = fontsize - 2)
        # plt.yticks(fontsize = fontsize - 2)
        # plt.xlabel('epoch #', fontsize = fontsize)
        # plt.ylabel('MSE loss', fontsize = fontsize)
        # plt.show()

        # plt.plot(np.arange(len(outputs[0])) * dt, outputs[0,:,0].detach().numpy(), label = 'learned')
        # plt.plot(np.arange(len(targets[0])) * dt, targets[0,:,0].detach().numpy(), label = 'target')
        # plt.plot(np.arange(len(initials[0])) * dt, initials[0,:,0].detach().numpy(), label = 'initial')
        # plt.xticks(fontsize = fontsize - 2)
        # plt.yticks(fontsize = fontsize - 2)
        # plt.xlabel('time (ms)', fontsize = fontsize)
        # plt.ylabel('firing rate', fontsize = fontsize)
        # plt.legend(fontsize = fontsize)
        # plt.show()

        # plt.plot(k_ms, label = 'k_m')
        # plt.plot(threshes, label = 'thresh')
        # plt.plot(asc_rs, label ='asc_r')
        # plt.plot(asc_ks, label = 'asc_k')
        # plt.plot(asc_amps, label = 'asc_amp')
        # plt.plot(weights, label = 'weight')
        # plt.xticks(fontsize = fontsize - 2)
        # plt.yticks(fontsize = fontsize - 2)
        # plt.xlabel('epoch #', fontsize = fontsize)
        # plt.ylabel('difference from target', fontsize = fontsize)
        # plt.legend()
        # plt.show()
        
        # quit()
        on_server = True
        main_name = "smnist_rnn_lastaspred_128unit_pixelbypixel"#"brnn200_noncued_moreascs_diffinit"#"brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_wtonly_agn_nodivstart"#lng_lngersim_uniformoffset_furthertrain"
        if on_server:
            base_name = main_name
            base_name_save =  main_name
            base_name_model = main_name
        else:
            base_name = "figures_wkof_053021/" + main_name
            base_name_save = "traininfo_wkof_053021/" + main_name
            base_name_model = "models_wkof_053021/" + main_name

        use_rnn = True
        hid_size = 128
        input_size = 1#28
        output_size = 10

        # # Generate freqs
        # num_freqs = 1
        # freq_min = 0.08#0.001
        # freq_max = 0.6

        # freqs = 10 ** np.linspace(np.log10(freq_min), np.log10(freq_max), num=num_freqs)

        # # Generate data
        # sim_time = 10
        # dt = 0.05
        # amp = 1
        # noise_mean = 0
        # noise_std = 0

        batch_size = 512

        # inputs, targets = ut.create_sines(sim_time, dt, amp, noise_mean, noise_std, freqs, input_size = input_size)
        # # inputs, _, _, targets = ut.generate_click_task_data(batch_size, seq_len, n_neuron, recall_duration, p_group, f0=0.5,
    # #                          n_cues=7, t_cue=100, t_interval=150,
    # #                          n_input_symbols=4)
        # traindataset = ut.create_dataset(inputs, targets, input_size)

        # # traindataset = ut.ThreeBitDataset(int(sim_time / dt), dataset_length=128)

        # # Generate model
        # delay = int(0.5 / dt)
        if use_rnn:
                model = RNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size)
        else:
                model = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size)
        # model.load_state_dict(torch.load("trained_model.pt"))#"saved_models/models_wkof_051621/brnn200_sussillo8_batched_hisgmav_predrive_scaleasc_wtonly_agn_nodivstart.pt"))
        # Train model
        num_epochs = 200
        lr = 1e-8#0.001#0.0025#0.0025#25#1#25
        reg_lambda = 1500

        # num_epochss = [200,100,50,10,1,1]
        training_info = ut.train_rbnn_mnist(model, batch_size, num_epochs, lr, not use_rnn, verbose = True)
        # training_info = ut.train_rbnn(model, traindataset, batch_size, num_epochs, lr, reg_lambda, glifr = not use_rnn)

        torch.save(model.state_dict(), "saved_models/" + base_name_model + ".pt")

        colors = ["sienna", "peru", "peachpuff", "salmon", "red", "darkorange", "purple", "fuchsia", "plum", "darkorchid", "slateblue", "mediumblue", "cornflowerblue", "skyblue", "aqua", "aquamarine", "springgreen", "green", "lightgreen"]

        # Plot outputs

        # ut.plot_predictions(model, int(sim_time / dt), batch_size)
        # plt.savefig("figures/" + base_name + "_final_outputs")
        # plt.close()


        # final_outputs = training_info["final_outputs"]

        # for i in range(num_freqs):
        #       plt.plot(np.arange(len(final_outputs[i][0])) * dt, final_outputs[i][0,:,0].detach().numpy(), c = colors[i], label=f"freq {freqs[i % len(colors)]}")
        #       plt.plot(np.arange(len(final_outputs[i][0])) * dt, targets[:, i], '--', c = colors[i % len(colors)])
        # # plt.legend()
        # plt.savefig("figures/" + base_name + "_final_outputs")
        # plt.close()

        # final_outputs_driven = training_info["final_outputs_driven"]
        # for i in range(num_freqs):
        #       plt.plot(np.arange(len(final_outputs_driven[i][0])) * dt, final_outputs_driven[i][0,:,0].detach().numpy(), c = colors[i], label=f"freq {freqs[i % len(colors)]}")
        #       plt.plot(np.arange(len(final_outputs_driven[i][0])) * dt, targets[:, i], '--', c = colors[i % len(colors)])
        # # plt.legend()
        # plt.xlabel("time (ms)")
        # plt.ylabel("firing rate (1/ms)")
        # plt.savefig("figures/" + base_name + "_final_outputs_driven")
        # plt.close()

        # init_outputs = training_info["init_outputs"]
        # for i in range(num_freqs):
        #       plt.plot(np.arange(len(init_outputs[i][0])) * dt, init_outputs[i][0,:,0].detach().numpy(), c = colors[i], label=f"freq {freqs[i % len(colors)]}")
        #       plt.plot(np.arange(len(init_outputs[i][0])) * dt, targets[:, i], '--', c = colors[i % len(colors)])
        # plt.legend()
        # plt.xlabel("time (ms)")
        # plt.ylabel("firing rate (1/ms)")
        # plt.savefig("figures/" + base_name + "_init_outputs")
        # plt.close()

        # init_outputs_driven = training_info["init_outputs_driven"]
        # for i in range(num_freqs):
        #       plt.plot(np.arange(len(init_outputs_driven[i][0])) * dt, init_outputs_driven[i][0,:,0].detach().numpy(), c = colors[i], label=f"freq {freqs[i % len(colors)]}")
        #       plt.plot(np.arange(len(init_outputs_driven[i][0])) * dt, targets[:, i], '--', c = colors[i % len(colors)])
        # plt.legend()
        # plt.xlabel("time (ms)")
        # plt.ylabel("firing rate (1/ms)")
        # plt.savefig("figures/" + base_name + "_init_outputs_driven")
        # plt.close()

        # Plot losses
        plt.plot(training_info["losses"])
        plt.xlabel("epoch #")
        plt.ylabel("loss")
        plt.savefig("figures/" + base_name + "_losses")
        plt.close()
        torch.save(training_info["losses"], "traininfo/" + base_name_save + "_losses.pt")

        colors = ["sienna", "gold", "chartreuse", "darkgreen", "lightseagreen", "deepskyblue", "blue", "darkorchid", "plum", "darkorange", "fuchsia", "tomato", "cyan", "greenyellow", "cornflowerblue", "limegreen", "springgreen", "green", "lightgreen", "aquamarine", "springgreen", "green", "lightgreen", "aquamarine", "springgreen", "green", "lightgreen", "aquamarine", "springgreen", "green", "lightgreen", "aquamarine", "springgreen", "green", "lightgreen"]

        if not use_rnn:
                i = -1
                for name in ["threshes", "k_ms", "asc_amps", "asc_rs", "asc_ks"]:
                        print(name)
                        i += 1
                        _, l = np.array(training_info[name]).shape
                        for j in range(l):
                                plt.plot(np.array(training_info[name])[:, j], color = colors[i], alpha = 0.5, label = name if j == 0 else "")
                plt.legend()
                plt.xlabel('epoch')
                plt.ylabel('parameter')
                plt.savefig("figures/" + base_name + "_parameters")
                plt.close()
        
        if not use_rnn:
                i = -1
                for name in ["thresh_grads", "k_m_grads", "asc_amp_grads", "asc_r_grads"]:
                        print(name)
                        i += 1
                        _, l = np.array(training_info[name]).shape
                        for j in range(l):
                                plt.plot(np.array(training_info[name])[:, j], color = colors[i], alpha = 0.5, label = name if j == 0 else "")
                plt.legend()
                plt.xlabel('epoch')
                plt.ylabel('parameter')
                plt.savefig("figures/" + base_name + "_parameter_grads")
                plt.close()
        
                # names = ["input_linear", "rec_linear", "output_linear"]
                # i = 0
                # for i in range(3):
                #       i += 1
                #       name = names[i]
                #       _, l = np.array(training_info["weights"][i]).shape
                #       for j in range(l):
                #               plt.plot(np.array(training_info["weights"][i])[:, j], color = colors[i], label = name if j == 0 else "")
                # plt.legend()
                # plt.xlabel('epoch')
                # plt.ylabel('parameter')
                # plt.savefig("figures/" + base_name + "_weights")
                # plt.close()
        
        # torch.save(training_info["losses"], "traininfo/" + base_name_save + "_losses.pt")
        with open("traininfo/" + base_name_save + ".pickle", 'wb') as handle:
                pickle.dump(training_info, handle)

if __name__ == '__main__':
        main()
