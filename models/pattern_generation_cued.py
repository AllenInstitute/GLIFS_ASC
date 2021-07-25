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
This file trains a network of rate-based GLIF neurons with after-spike currents on a cued pattern generation task.

Trained model is saved to the folder specified by saved_models/base_name_model
Figures on learned outputs, parameters, weights, gradients, and losses over training are saved to the folder specified by figures/base_name
Loss array and model information saved in folder specified by traininfo/base_name_save
"""

def main():
        main_name = "brnn-delay_cued_10ms_64units_4d_lateralconns"#"rnn-nodelay_cued_10ms_128units_10d"
        base_name = "figures_wkof_071821/" + main_name
        base_name_save = "traininfo_wkof_071821/" + main_name
        base_name_model = "models_wkof_071821/" + main_name

        # Model dimensions
        use_rnn = False
        use_lstm = False

        hid_size = 64#64#45
        input_size = 20#8
        output_size = 1

        # Target specifications
        sim_time = 10
        dt = 0.05
        amp = 1
        noise_mean = 0
        noise_std = 0

        batch_size = 1
        num_freqs = 4
        freq_min = 0.08#0.001
        freq_max = 0.6

        # Training specifications
        num_epochs = 500
        lr = 0.00005#0.005
        reg_lambda = 0#1500
        decay = False

        # Generate freqs
        freqs = 10 ** np.linspace(np.log10(freq_min), np.log10(freq_max), num=num_freqs)

        inputs, targets = utt.create_sines_cued(sim_time, dt, amp, noise_mean, noise_std, freqs, input_size)
        traindataset = utt.create_dataset(inputs, targets)

        # # traindataset = ut.ThreeBitDataset(int(sim_time / dt), dataset_length=128)

        # # Generate model
        if use_rnn:
                model = RNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, dt=dt)
        else:
                model = BNNFC(in_size = input_size, hid_size = hid_size, out_size = output_size, dt=dt)

        # Train model
        training_info = ut.train_rbnn(model, traindataset, batch_size, num_epochs, lr, reg_lambda, glifr = not use_rnn, task = "pattern", decay=decay)

        torch.save(model.state_dict(), "saved_models/" + base_name_model + ".pt")

        colors = ["sienna", "peru", "peachpuff", "salmon", "red", "darkorange", "purple", "fuchsia", "plum", "darkorchid", "slateblue", "mediumblue", "cornflowerblue", "skyblue", "aqua", "aquamarine", "springgreen", "green", "lightgreen"]

        # Plot outputs
        final_outputs = training_info["final_outputs"]

        for i in range(num_freqs):
                plt.plot(np.arange(len(final_outputs[i][0,:])) * dt, final_outputs[i][0,:].detach().numpy(), c = colors[i], label=f"freq {freqs[i % len(colors)]}")
                plt.plot(np.arange(len(final_outputs[i][0,:])) * dt, targets[:,i], '--', c = colors[i % len(colors)])
        # plt.legend()
        plt.xlabel("time (ms)")
        plt.ylabel("firing rate (1/ms)")
        plt.savefig("figures/" + base_name + "_final_outputs")
        plt.close()


        init_outputs = training_info["init_outputs"]
        for i in range(num_freqs):
                plt.plot(np.arange(len(init_outputs[i][0,:])) * dt, init_outputs[i][0,:].detach().numpy(), c = colors[i], label=f"freq {freqs[i % len(colors)]}")
                plt.plot(np.arange(len(init_outputs[i][0,:])) * dt, targets[:, i], '--', c = colors[i % len(colors)])
        # plt.legend()
        plt.xlabel("time (ms)")
        plt.ylabel("firing rate (1/ms)")
        plt.savefig("figures/" + base_name + "_init_outputs")
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
        with open("traininfo/" + base_name_save + ".pickle", 'wb') as handle:
                pickle.dump(training_info, handle)

if __name__ == '__main__':
        main()
