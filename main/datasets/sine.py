import torch
import torch.utils.data as tud
import numpy as np
import pytorch_lightning as pl
from utils.check import check_nonnegative_float, check_nonnegative_int, check_positive_float, check_positive_int
# torch.autograd.set_detect_anomaly(True)

class SineDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.dataset = create_sine_dataset(
            sim_time = self.hparams.sim_time, 
            dt = self.hparams.dt,
            num_freqs = self.hparams.num_freqs, 
            freq_min = self.hparams.freq_min,
            freq_max = self.hparams.freq_max,
            amp = self.hparams.amp,
            noise_mean = self.hparams.noise_mean,
            noise_std = self.hparams.noise_std
        )

    def train_dataloader(self):
        return tud.DataLoader(self.dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return tud.DataLoader(self.dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return tud.DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=self.hparams.num_workers)

    @staticmethod
    def add_sine_args(parent_parser):
        parser = parent_parser.add_argument_group("sine")
        parser.add_argument("--sim_time", type=check_positive_float, default=5)
        parser.add_argument("--num_freqs", type=check_positive_int, default=6)
        parser.add_argument("--freq_min", type=check_positive_float, default=0.08)
        parser.add_argument("--freq_max", type=check_positive_float, default=0.6)
        parser.add_argument("--amp", type=float, default=1)
        parser.add_argument("--noise_mean", type=float, default=0)
        parser.add_argument("--noise_std", type=float, default=0)

def create_sine_dataset(
    sim_time = 5,
    dt=0.05,
    num_freqs = 6,
    freq_min = 0.08,
    freq_max = 0.6,
    amp = 1,
    noise_mean = 0,
    noise_std = 0,
):
    freqs = 10 ** np.linspace(np.log10(freq_min), np.log10(freq_max), num=num_freqs)
    
    # Generate the actual sine waves/inputs
    inputs, targets = create_sines_amp(sim_time, dt, amp, noise_mean, noise_std, freqs)

    # Convert into dataset
    nsteps, n, input_size = inputs.shape

    inputs = np.moveaxis(inputs, 0,1)
    targets = np.moveaxis(targets, 0,1)

    inputs = inputs.reshape((n, nsteps, input_size))
    targets = targets.reshape((n, nsteps, -1))

    return tud.TensorDataset(torch.tensor(inputs).float(), torch.tensor(targets).float())

def create_sines_amp(sim_time, dt, amp, noise_mean, noise_std, freqs, **kwargs):
    """
    Create a dataset of different frequency sinusoids adapted  and simplified from
    (David Sussillo, Omri Barak, 2013) pattern generation task
    Parameters
    ----------
    sim_time : float
        number of ms of total simulation time
    dt : float
        number of ms in each timestep
    amp : float
        amplitude of the sinusoid
    noise_mean : float
        mean of noise added to sinusoid
    noise_std : float
        std of noise added to sinusoid
    freqs : List
        list of frequencies (1/ms or kHz) of sinusoids
    Returns
    -------
    Numpy Array(nsteps, len(freqs))
        input sequences (all 1-D)
    Numpy Arryay(nsteps, len(freqs))
        target sequences (all 1-D)
    """
    n = len(freqs)
    nsteps = int(sim_time / dt)
    time = np.arange(start = 0, stop = sim_time, step = dt)

    targets = np.empty((nsteps, n))
    inputs = np.empty((nsteps, n))

    for i in range(n):
        offset = (i / n) + 0.25
        noise = np.random.normal(noise_mean, noise_std, nsteps)
        freq = 2 * np.pi * freqs[i]
        
        targets[:, i] = amp * np.sin(freq * time) + noise + offset
        inputs[:,i] = offset

    inputs = np.expand_dims(inputs, -1)
    targets = np.expand_dims(targets, -1)

    return inputs, targets