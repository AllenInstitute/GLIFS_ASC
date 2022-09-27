import os

import numpy as np
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

class MetricTracker(Callback):
    # Regarding training
    def on_fit_start(self, trainer, pl_module):
        self.train_loss = []
        self.train_acc = []
    
    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss.append(trainer.logged_metrics["train_loss_epoch"].item())
        if "train_acc_epoch" in trainer.logged_metrics:
            self.train_acc.append(trainer.logged_metrics["train_acc_epoch"].item())
    
    def on_fit_end(self, trainer, pl_module):
        train_loss = np.asarray(self.train_loss).reshape(-1, 1)
        train_loss.tofile(os.path.join(trainer.log_dir, "train_loss.csv"), sep=',')
        if "train_acc_epoch" in trainer.logged_metrics:
            train_acc = np.asarray(self.train_acc).reshape(-1, 1)
            train_acc.tofile(os.path.join(trainer.log_dir, "train_acc.csv"), sep=',')
    
    # Regarding testing
    def on_test_start(self, trainer, pl_module):
        self.test_loss = []
        self.test_acc = []
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.test_loss.append(trainer.logged_metrics["test_loss_epoch"].item())
        if "test_acc_epoch" in trainer.logged_metrics:
            self.test_acc.append(trainer.logged_metrics["test_acc_epoch"].item())
    
    def on_test_end(self, trainer, pl_module):
        test_loss = np.asarray(self.test_loss).reshape(-1, 1)
        test_loss.tofile(os.path.join(trainer.log_dir, "test_loss.csv"), sep=',')
        if "test_acc_epoch" in trainer.logged_metrics:
            test_acc = np.asarray(self.test_acc).reshape(-1, 1)
            test_acc.tofile(os.path.join(trainer.log_dir, "test_acc.csv"), sep=',')

class ParameterTracker(Callback):
    # def on_fit_start(self, trainer, pl_module):
    #     learnable_params = pl_module.learnable_params_()
    #     for n, p in pl_module.named_parameters():
    #         if n in learnable_params and "weight" not in n:
    #             with open(os.path.join(trainer.log_dir, f"{n.split('.')[-1]}_param-trajectory.csv"), 'w') as f:
    #                 pass
            
    def on_train_epoch_end(self, trainer, pl_module):
        learnable_params = pl_module.learnable_params_()
        for n, p in pl_module.named_parameters():
            if n in learnable_params and "weight" not in n:
                with open(os.path.join(trainer.log_dir, f"{n.split('.')[-1]}_param-trajectory.csv"), 'a') as f:
                    np.savetxt(f, p.cpu().detach().numpy().reshape((1, -1)), delimiter=',', newline='\n')

def create_model_checkpoint():
    return ModelCheckpoint(save_last=True)

class SigmaVAnneal(Callback):
    def on_fit_start(self, trainer, pl_module):
        self.sigma_vs = np.linspace(1e-3, 1, num=trainer.max_epochs)
    
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.neuron_layer.sigma_v = self.sigma_vs[-1 - trainer.current_epoch]