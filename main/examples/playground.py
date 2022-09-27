from argparse import ArgumentParser
import numpy as np
import os
import torch
import math

from models.pl_modules import add_structure_args, add_general_model_args, add_training_args, GLIFRN, RNN, LSTMN

def plot_example_step():
    # Simulates single neuron responding to step-like current
    filename = "lmnist-sample-outputs-step-cluster3-idx7"

    params = {
        "lmnist-sample-outputs-step-cluster0-idx1": {
            "asc_r": (4.161372184753417969e-01,4.154667854309082031e-01),
            "asc_amp": (7.761620879173278809e-01,7.741597294807434082e-01),
            "asc_k": (1.024937272071838379e+00,1.026827335357666016e+00),
            "thresh": 1.027656435966491699e+00,
            "k_m": 7.396198064088821411e-02
        },
        "lmnist-sample-outputs-step-cluster1-idx2": {
            "asc_r": (7.544246912002563477e-01,7.376768589019775391e-01),
            "asc_amp": (-1.352998733520507812e+00,-1.286132216453552246e+00),
            "asc_k": (4.560432434082031250e-01,5.001413822174072266e-01),
            "thresh": 6.666137576103210449e-01,
            "k_m":4.485654458403587341e-02
        },
        "lmnist-sample-outputs-step-cluster2-idx0": {
            "asc_r": (-3.078615665435791016e-02,-5.044603347778320312e-02),
            "asc_amp": (6.649822741746902466e-02,6.622596085071563721e-02),
            "asc_k": (2.321566581726074219e+00,2.340293407440185547e+00),
            "thresh": 4.252734780311584473e-01,
            "k_m": 1.864943094551563263e-02
        },
        "lmnist-sample-outputs-step-cluster3-idx4": {
            "asc_r": (3.082646727561950684e-01,2.363232970237731934e-01),
            "asc_amp": (-3.706759810447692871e-01,-3.146711587905883789e-01),
            "asc_k": (1.448156476020812988e+00,1.550688028335571289e+00),
            "thresh": 8.086432218551635742e-01,
            "k_m": 3.169588744640350342e-02
        },
        "lmnist-sample-outputs-step-cluster0-idx34": {
            "asc_r": (5.128648281097412109e-01,5.015183687210083008e-01),
            "asc_amp": (8.973875641822814941e-01,8.850873708724975586e-01),
            "asc_k": (8.435172438621520996e-01,8.612868189811706543e-01),
            "thresh": 7.547109127044677734e-01,
            "k_m": 4.605109617114067078e-02
        },
        "lmnist-sample-outputs-step-cluster1-idx10": {
            "asc_r": (4.946979880332946777e-01,4.870920181274414062e-01),
            "asc_amp": (-9.102169871330261230e-01,-9.110727310180664062e-01),
            "asc_k": (8.446884751319885254e-01,8.440377712249755859e-01),
            "thresh": 1.000137925148010254e+00,
            "k_m":5.766574293375015259e-02
        },
        "lmnist-sample-outputs-step-cluster2-idx5": {
            "asc_r": (7.447159290313720703e-02,1.598899364471435547e-01),
            "asc_amp": (-1.090515181422233582e-01,-1.521869152784347534e-01),
            "asc_k": (1.869365692138671875e+00,1.736939072608947754e+00),
            "thresh": 6.099369525909423828e-01,
            "k_m": 1.928066276013851166e-02
        },
        "lmnist-sample-outputs-step-cluster3-idx7": {
            "asc_r": (6.159752607345581055e-02,6.507426500320434570e-02),
            "asc_amp": (-2.279593646526336670e-01,-2.295845896005630493e-01),
            "asc_k": (1.532290220260620117e+00,1.530838489532470703e+00),
            "thresh": 8.167200088500976562e-01,
            "k_m": 3.225067630410194397e-02
        }
    }
    # filename_dir = "results_wkof_080821/" + filename
    name = filename

    sim_time = 40
    dt = 0.05
    nsteps = int(sim_time / dt)

    input_size = 1
    output_size = 1
    hid_size = 1

    inputs = 0.001 * torch.ones(1, nsteps, input_size)
    # inputs[:, 100:400, :] = 1

    parser = ArgumentParser()
    # Model-specific
    add_structure_args(parser)
    add_general_model_args(parser)
    GLIFRN.add_model_specific_args(parser)

    # add model specific args
    args = parser.parse_args()
    args.__dict__["input_size"] = input_size
    args.__dict__["hidden_size"] = hid_size
    args.__dict__["output_size"] = output_size
    args.__dict__["dt"] = dt
    args.__dict__["output_weight"] = False

    model_glif = GLIFRN(**vars(args))
    asc_r = params[filename]["asc_r"]
    asc_amp = params[filename]["asc_amp"]
    asc_k = params[filename]["asc_k"]
    thresh = params[filename]["thresh"]
    k_m = params[filename]["k_m"]
    
    model_glif.reset_state(1)
    with torch.no_grad():
        model_glif.neuron_layer.weight_iv.data = torch.ones((input_size, hid_size))
        model_glif.neuron_layer.weight_lat.data = torch.zeros((hid_size, hid_size))
        model_glif.neuron_layer.thresh.data[0,0] = thresh

        model_glif.neuron_layer.trans_k_m[0,0] = math.log(k_m * dt / (1 - (k_m * dt))) 

        asc_r1, asc_r2 = asc_r
        asc_amp1, asc_amp2 = asc_amp
        asc_k1, asc_k2 = asc_k

        model_glif.neuron_layer.trans_r_j[0,0,0] = math.log((1 - asc_r1) / (1 + asc_r1))
        model_glif.neuron_layer.trans_r_j[1,0,0] = math.log((1 - asc_r2) / (1 + asc_r2))

        model_glif.neuron_layer.a_j[0,0,0] = asc_amp1
        model_glif.neuron_layer.a_j[1,0,0] = asc_amp2

        model_glif.neuron_layer.trans_k_j[0,0,0] = math.log(asc_k1 * dt / (1 - (asc_k1 * dt))) 
        model_glif.neuron_layer.trans_k_j[1,0,0] = math.log(asc_k2 * dt / (1 - (asc_k2 * dt))) 

    outputs, voltages, ascs, syns = model_glif.forward(inputs, track=True)
    outputs = outputs.detach().numpy()
    voltages = voltages.detach().numpy()
    ascs = ascs.detach().numpy()
    syns = syns.detach().numpy()

    np.savetxt(f"./results/playground/{name}.csv", outputs[0,:,0], delimiter=",")
    np.savetxt(f"./results/playground/{name}-syn.csv", syns[0,:,0], delimiter=",")
    np.savetxt(f"./results/playground/{name}-voltage.csv", voltages[0,:,0], delimiter=",")
    np.savetxt(f"./results/playground/{name}-asc.csv", ascs[:,0,:,0], delimiter=",")
    np.savetxt(f"./results/playground/{name}-in.csv", inputs[0,:,0], delimiter=",")

    # plt.plot(outputs[0,:,0], label=name)
    # plt.legend()
    # plt.show()

def glifr_homa():
    model = GLIFRN.load_from_checkpoint("./results/sine/glifr_homa/trial_0/lightning_logs/version_0/checkpoints/last.ckpt")
    # learnable_params = pl_module.learnable_params_()
    # for n, p in model.named_parameters():
    #     if "weight" not in n:
    #         with open(os.path.join(f"lmnist_glifr_homa_{n.split('.')[-1]}_param-trajectory.csv"), 'a') as f:
    #             np.savetxt(f, p.cpu().detach().numpy().reshape((1, -1)), delimiter=',', newline='\n')

    ficurve_simtime = 5
    # Produces firing rates and input currents needed for a f-I curve
    sim_time = ficurve_simtime
    dt = model.hparams.dt
    nsteps = int(sim_time / dt)

    i_syns = np.arange(-10000, 10000, step=100)

    input = torch.zeros(1, nsteps, model.hparams.input_size)

    f_rates = np.zeros((len(i_syns), model.hparams.hidden_size))
    for i in range(len(i_syns)):
        print(f"{i}/{len(i_syns)}")
        firing = torch.zeros((input.shape[0], model.hparams.hidden_size))
        voltage = torch.zeros((input.shape[0], model.hparams.hidden_size))
        syncurrent = torch.zeros((input.shape[0], model.hparams.hidden_size))
        ascurrents = torch.zeros((2, input.shape[0], model.hparams.hidden_size))
        outputs_temp = torch.zeros(1, nsteps, model.hparams.hidden_size)

        firing_delayed = torch.zeros((input.shape[0], nsteps, model.hparams.hidden_size))

        model.neuron_layer.I0 = i_syns[i]
        for step in range(nsteps):
            x = input[:, step, :]
            firing, voltage, ascurrents, syncurrent = model.neuron_layer(x, firing, voltage, ascurrents, syncurrent, firing_delayed[:, step, :])
            outputs_temp[0, step, :] = firing
        f_rates[i, :] = torch.mean(outputs_temp, 1).detach().numpy().reshape((1, -1))

    print(f"f_rates.shape = {f_rates.shape}")

    slopes = []
    for i in range(model.hparams.hidden_size):
        i_syns_these = i_syns
        f_rates_these = f_rates[:,i]
        indices = np.logical_not(np.logical_or(np.isnan(i_syns_these), np.isnan(f_rates_these)))     
        indices = np.array(indices)
        i_syns_these = i_syns_these[indices]
        
        f_rates_these = f_rates_these[indices] #* sim_time / dt
        i_syns_these = i_syns_these[f_rates_these > 0.01]
        f_rates_these = f_rates_these[f_rates_these > 0.01] * sim_time / dt


        # A = np.vstack([i_syns_these, np.ones_like(i_syns_these)]).T
        # m, c = np.linalg.lstsq(A, f_rates_these)[0]
        # if len(f_rates_these) > 0:
        #     slopes.append(m)

        # if m < 0:
        #     print(f"found negative slope in neuron {i}")
        #     print(f_rates_these)
        #     print(m)
        # plt.plot(i_syns_these, f_rates_these)
    # np.savetxt(os.path.join(self.trainer.logger.log_dir, "ficurve_slopes.csv"), np.array(slopes), delimiter=",")
    np.savetxt("sine_glifr_homa_isyns.csv", np.array(i_syns), delimiter=",")
    np.savetxt("sine_glifr_homa_frates.csv", np.array(f_rates), delimiter=",")


if __name__ == '__main__':
    glifr_homa()
    # plot_example_step()