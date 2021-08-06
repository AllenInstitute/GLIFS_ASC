import matplotlib
matplotlib.use('Agg')

# author: @chloewin
# 03/07/21
import pickle
import datetime
import utils as ut
import utils_task as utt
from networks import RNNFC, BNNFC

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
        main_name = "amplitude-2sines_rnn-128units-newinit-sgd"
        base_name = "figures_wkof_072521/" + main_name
        base_name_save = "traininfo_wkof_072521/" + main_name
        base_name_model = "models_wkof_072521/" + main_name

 # Model dimensions
        use_rnn = True
        use_lstm = False
        initburst = False

        hid_size = 128#512
        input_size = 1
        output_size = 1
        sparseness = 0

        # Target specifications
        sim_time = 5
        dt = 0.05
        amp = 1
        noise_mean = 0
        noise_std = 0

        batch_size = 2
        num_freqs = 2
        freq_min = 0.08#0.001
        freq_max = 0.6

        # Training specifications
        num_epochs = 10000#5000
        lr = 0.01#0.0001#0.005
        decay = False
        sgd = True

        # Generate freqs
        freqs = 10 ** np.linspace(np.log10(freq_min), np.log10(freq_max), num=num_freqs)

        # Generate data
        inputs, targets = utt.create_sines_amp(sim_time, dt, amp, noise_mean, noise_std, freqs)
        traindataset = utt.create_dataset(inputs, targets)

        # # Generate model
        if use_rnn:
                if use_lstm:
                        model = RNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, dt=dt)
                else:
                        model = RNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, dt=dt, sparseness=sparseness)
        else:
                model = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, dt=dt, initburst=initburst, sparseness=sparseness)

        # Train model
        training_info = ut.train_rbnn(model, traindataset, batch_size, num_epochs, lr, glifr = not use_rnn, task = "pattern", decay=decay, sgd=sgd)

        torch.save(model.state_dict(), "saved_models/" + base_name_model + ".pt")

        colors = ["sienna", "peru", "peachpuff", "salmon", "red", "darkorange", "purple", "fuchsia", "plum", "darkorchid", "slateblue", "mediumblue", "cornflowerblue", "skyblue", "aqua", "aquamarine", "springgreen", "green", "lightgreen"]

        # Plot outputs

        #ut.plot_predictions(model, int(sim_time / dt), batch_size)
        #plt.savefig("figures/" + base_name + "_final_outputs")
        #plt.close()


        final_outputs = training_info["final_outputs"]

        for i in range(num_freqs):
                print(len(final_outputs))
                plt.plot(np.arange(len(final_outputs[i][0])) * dt, final_outputs[i][0,:,0].detach().numpy(), c = colors[i % len(colors)], label=f"freq {freqs[i]}")
                plt.plot(np.arange(len(final_outputs[i][0])) * dt, targets[:, i], '--', c = colors[i % len(colors)])
        # # plt.legend()
        plt.savefig("figures/" + base_name + "_final_outputs")
        plt.close()

        # final_outputs_driven = training_info["final_outputs_driven"]
        # for i in range(num_freqs):
        #       plt.plot(np.arange(len(final_outputs_driven[i][0])) * dt, final_outputs_driven[i][0,:,0].detach().numpy(), c = colors[i], label=f"freq {freqs[i % len(colors)]}")
        #       plt.plot(np.arange(len(final_outputs_driven[i][0])) * dt, targets[:, i], '--', c = colors[i % len(colors)])
        # # plt.legend()
        # plt.xlabel("time (ms)")
        # plt.ylabel("firing rate (1/ms)")
        # plt.savefig("figures/" + base_name + "_final_outputs_driven")
        # plt.close()

        init_outputs = training_info["init_outputs"]
        for i in range(num_freqs):
                plt.plot(np.arange(len(init_outputs[i][0])) * dt, init_outputs[i][0,:,0].detach().numpy(), c = colors[i % len(colors)], label=f"freq {freqs[i]}")
                plt.plot(np.arange(len(init_outputs[i][0])) * dt, targets[:, i], '--', c = colors[i % len(colors)])
        # plt.legend()
        plt.xlabel("time (ms)")
        plt.ylabel("firing rate (1/ms)")
        plt.savefig("figures/" + base_name + "_init_outputs")
        plt.close()

        final_outputs_driven = training_info["final_outputs_driven"]
        for i in range(num_freqs):
                plt.plot(np.arange(len(final_outputs_driven[i][0])) * dt, final_outputs_driven[i][0,:,0].detach().numpy(), c = colors[i % len(colors)], label=f"freq {freqs[i]}")
                plt.plot(np.arange(len(final_outputs_driven[i][0])) * dt, targets[:, i], '--', c = colors[i % len(colors)])
        # plt.legend()
        plt.xlabel("time (ms)")
        plt.ylabel("firing rate (1/ms)")
        plt.savefig("figures/" + base_name + "_final_outputs_driven")
        plt.close()

        init_outputs = training_info["init_outputs"]
        for i in range(num_freqs):
                plt.plot(np.arange(len(init_outputs[i][0])) * dt, init_outputs[i][0,:,0].detach().numpy(), c = colors[i % len(colors)], label=f"freq {freqs[i]}")
                plt.plot(np.arange(len(init_outputs[i][0])) * dt, targets[:, i], '--', c = colors[i % len(colors)])
        plt.legend()
        plt.xlabel("time (ms)")
        plt.ylabel("firing rate (1/ms)")
        plt.savefig("figures/" + base_name + "_init_outputs")
        plt.close()

        init_outputs_driven = training_info["init_outputs_driven"]
        for i in range(num_freqs):
                plt.plot(np.arange(len(init_outputs_driven[i][0])) * dt, init_outputs_driven[i][0,:,0].detach().numpy(), c = colors[i % len(colors)], label=f"freq {freqs[i]}")
                plt.plot(np.arange(len(init_outputs_driven[i][0])) * dt, targets[:, i], '--', c = colors[i % len(colors)])
        plt.legend()
        plt.xlabel("time (ms)")
        plt.ylabel("firing rate (1/ms)")
        plt.savefig("figures/" + base_name + "_init_outputs_driven")
        plt.close()

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
                for name in ["asc_amps", "asc_rs", "asc_ks", "threshes", "k_ms", "weights"]:
                        print(name)
                        i += 1
                        _, l = np.array(training_info[name]).shape
                        for j in range(l):
                            plt.plot(np.array(training_info[name])[:,j], color = colors[i], alpha = 0.5, label = name if j == 0 else "")
                # plt.legend()
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
                # plt.legend()
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
