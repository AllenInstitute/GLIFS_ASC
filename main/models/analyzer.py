import numpy as np
import os
import torch

class NetworkAnalyzer():
    def __init__(self, trainer, pl_module, data_module):
        self.trainer = trainer
        self.model = pl_module
        self.data_module = data_module
        # self.prev_log_dir = prev_log_dir

    def run_ablation_experiment(self, pcts, ntrials):
        """
        Runs ablation experiment on network.
        (i.e., will, for each provided percentage, run ntrials
        trials where in a trial, a randomly chosen proportion of
        the neurons are silenced, and the testing accuracy is computed.)

        Saves results to a CSV which is stored in the provided
        trainer's log_dir.
        """
        accs = np.zeros((len(pcts), ntrials))
        losses = np.zeros((len(pcts), ntrials))
        for pidx in range(len(pcts)):
            pct = pcts[pidx]
            for tidx in range(ntrials):
                idx = np.random.choice(self.model.hparams.hidden_size, int(pct * self.model.hparams.hidden_size), replace=False)
                self.model.silence(idx)
                test_metrics = self.trainer.test(datamodule=self.data_module, model = self.model)
                accs[pidx, tidx] = 0 if "test_acc_epoch" not in test_metrics[0] else test_metrics[0]["test_acc_epoch"]
                losses[pidx, tidx] = test_metrics[0]["test_loss_epoch"]
        # print(losses.shape)
        np.savetxt(os.path.join(self.trainer.logger.log_dir, "ablation_losses.csv"), losses, delimiter=",")
        np.savetxt(os.path.join(self.trainer.logger.log_dir, "ablation_accs.csv"), accs, delimiter=",")

    # @staticmethod
    # def run_dropout_experiment(args):

    def test_responses(self):
        """
        Record responses of trained network to data in the
        testing dataloader in the trainer, along with the targets.
        """
        outputs_all = []
        targets_all = []
        self.model.eval()
        with torch.no_grad():
            test_loader = self.data_module.test_dataloader()
            for batch_idx, batch in enumerate(test_loader):
                x, targets = batch
                self.model.reset_state(len(targets))
                outputs = self.model(x)
                # print(outputs)
                outputs_all.append(outputs)
                targets_all.append(targets)
        outputs_all = torch.stack(outputs_all).detach().numpy().reshape((6, -1))
        targets_all = torch.stack(targets_all).detach().numpy().reshape((6, -1))
        # print(targets_all.shape)
        os.makedirs(self.trainer.logger.log_dir, exist_ok=True)
        np.savetxt(os.path.join(self.trainer.logger.log_dir, "test_outputs.csv"), outputs_all, delimiter=",")
        np.savetxt(os.path.join(self.trainer.logger.log_dir, "test_targets.csv"), targets_all, delimiter=",")
        # outputs_all.tofile(os.path.join(self.trainer.logger.log_dir, f"test_outputs.csv"), sep=',')
        # targets_all.tofile(os.path.join(self.trainer.logger.log_dir, f"test_targets.csv"), sep=',')

    def test_step_responses(self):
        """
        Record responses of trained network to constant input signals.
        """
        outputs_all = []
        targets_all = []
        sim_time = 10
        nsteps = int(sim_time / self.model.hparams.dt)
        self.model.eval()
        with torch.no_grad():
            for i in np.arange(0, 50, 10):
                x = i * np.ones()
                x, targets = batch
                self.model.reset_state(len(targets))
                outputs = self.model(x)
                outputs_all.append(outputs)
                targets_all.append(targets)
        outputs_all = torch.stack(outputs_all).detach().numpy().reshape((6, -1))
        targets_all = torch.stack(targets_all).detach().numpy().reshape((6, -1))
        # print(targets_all.shape)
        os.makedirs(self.trainer.logger.log_dir, exist_ok=True)
        np.savetxt(os.path.join(self.trainer.logger.log_dir, "test_outputs.csv"), outputs_all, delimiter=",")
        np.savetxt(os.path.join(self.trainer.logger.log_dir, "test_targets.csv"), targets_all, delimiter=",")
        # outputs_all.tofile(os.path.join(self.trainer.logger.log_dir, f"test_outputs.csv"), sep=',')
        # targets_all.tofile(os.path.join(self.trainer.logger.log_dir, f"test_targets.csv"), sep=',')

    def save_parameters(self):
        # if "glifr" not in self.trainer.train_params.network_type:
        #     raise NotImplementedError
        parameters = np.zeros((self.model.hparams.hidden_size, 8))
        parameters[:, 0] = self.model.neuron_layer.thresh.detach().numpy().reshape(-1)
        parameters[:, 1] = self.model.neuron_layer.transform_to_k(self.model.neuron_layer.trans_k_m).detach().numpy().reshape(-1)
        parameters[:, 2] = self.model.neuron_layer.transform_to_asc_r(self.model.neuron_layer.trans_r_j).detach().numpy()[0,0,:]
        parameters[:, 3] = self.model.neuron_layer.transform_to_asc_r(self.model.neuron_layer.trans_r_j).detach().numpy()[1,0,:]
        parameters[:, 4] = self.model.neuron_layer.transform_to_k(self.model.neuron_layer.trans_k_j).detach().numpy()[0,0,:]
        parameters[:, 5] = self.model.neuron_layer.transform_to_k(self.model.neuron_layer.trans_k_j).detach().numpy()[1,0,:]
        parameters[:, 6] = self.model.neuron_layer.a_j.detach().numpy()[0,0,:]
        parameters[:, 7] = self.model.neuron_layer.a_j.detach().numpy()[1,0,:]
        np.savetxt(os.path.join(self.trainer.logger.log_dir, f"learnedparams.csv"), parameters, delimiter=',')

    def saliency(self):
        pass
        # output_max = output[0, predicted_discrete]
        # output_max.backward()
        # saliency = images.grad.data.abs()
        # saliency -= saliency.min(1, keepdim=True)[0]
        # saliency /= saliency.max(1, keepdim=True)[0]
        # print(saliency)
        # save_image(saliency[0], f"figures/smnist_saliency/glifr_lheta_{label}.png")

    def plot_ficurve(self):
        ficurve_simtime = 5
        # Produces firing rates and input currents needed for a f-I curve
        sim_time = ficurve_simtime
        dt = 0.05
        nsteps = int(sim_time / dt)

        i_syns = np.arange(-10000, 10000, step=100)

        input = torch.zeros(1, nsteps, self.model.hparams.input_size)

        f_rates = np.zeros((len(i_syns), self.model.hparams.hidden_size))
        for i in range(len(i_syns)):
            firing = torch.zeros((input.shape[0], self.model.hparams.hidden_size))
            voltage = torch.zeros((input.shape[0], self.model.hparams.hidden_size))
            syncurrent = torch.zeros((input.shape[0], self.model.hparams.hidden_size))
            ascurrents = torch.zeros((2, input.shape[0], self.model.hparams.hidden_size))
            outputs_temp = torch.zeros(1, nsteps, self.model.hparams.hidden_size)

            firing_delayed = torch.zeros((input.shape[0], nsteps, self.model.hparams.hidden_size))

            self.model.neuron_layer.I0 = i_syns[i]
            for step in range(nsteps):
                x = input[:, step, :]
                firing, voltage, ascurrents, syncurrent = self.model.neuron_layer(x, firing, voltage, ascurrents, syncurrent, firing_delayed[:, step, :])
                outputs_temp[0, step, :] = firing
            f_rates[i, :] = torch.mean(outputs_temp, 1).detach().numpy().reshape((1, -1))

        print(f"f_rates.shape = {f_rates.shape}")

        slopes = []
        for i in range(self.model.hparams.hidden_size):
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
        np.savetxt(os.path.join(self.trainer.logger.log_dir, "isyns.csv"), np.array(i_syns), delimiter=",")
        np.savetxt(os.path.join(self.trainer.logger.log_dir, "frates.csv"), np.array(f_rates), delimiter=",")

