import os
import time
import numpy as np
from torch.utils import data
from datasets.nmnist_parser import readAERFile
import torch
import torch.utils.data as tud
from utils.check import check_nonnegative_float, check_nonnegative_int, check_positive_float, check_positive_int
import pytorch_lightning as pl
class NMNISTDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_set, self.test_set = nmnist_train_test(self.hparams.nmnist_path, self.hparams.num_timesteps, self.hparams.duration_timestep)

    def train_dataloader(self):
        return tud.DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return tud.DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        return tud.DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)

    @staticmethod
    def add_nmnist_args(parent_parser):
        parser = parent_parser.add_argument_group("nmnist")
        parser.add_argument("--nmnist_path", type=str)
        parser.add_argument("--num_timesteps", type=check_positive_int, default=15)
        parser.add_argument("--duration_timestep", type=check_positive_float)

def nmnist_train_test(path, num_timesteps, duration_timestep):
    train_dataset = NMnist(path=path, num_timesteps=num_timesteps, duration_timestep=duration_timestep, is_train=True)
    test_dataset = NMnist(path=path, num_timesteps=num_timesteps, duration_timestep=duration_timestep, is_train=False)
    return train_dataset, test_dataset

class NMnist(data.Dataset):
    """
    NMnist dataset from
    Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N.
    “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades",
    Frontiers in Neuroscience, vol.9, no.437, Oct. 2015
    Available for download: https://www.garrickorchard.com/datasets/n-mnist
    """

    def __init__(self, path, num_timesteps, duration_timestep, is_train: bool = True, transforms=None, **kwargs):
        # path: str, num_timesteps = 15, duration_timestep = 1
        # path = args.path
        # num_timesteps = args.num_timesteps
        # duration_timestep = args.duration_timestep

        if not os.path.exists(path) or len(os.listdir(path)) == 0:
            raise "Data not found at path %s" % path
        
        self.num_timesteps = num_timesteps
        self.duration_timestep = duration_timestep

        path = os.path.join(path, "Train" if is_train else "Test")

        self._files = []
        self._labels = []

        for root, dirs, files in os.walk(path):
            digit = os.path.basename(root)
            for file in files:
                if file.endswith(".bin"):
                    self._files.append(os.path.join(root, file))
                    self._labels.append(int(digit))

        self._files = np.asarray(self._files)
        self._labels = np.asarray(self._labels)
        self.transforms = transforms

    def __len__(self):
        return self._files.size

    def __getitem__(self, index):
        spike_train = readAERFile(self._files[index])
        spike_train.width = 34
        spike_train.height = 34
        spike_train.duration = spike_train.ts.max() + 1
        if self.transforms is not None:
            spike_train = self.transforms(spike_train)
        num_timesteps = self.num_timesteps
        frame = torch.zeros((num_timesteps, 2, 34, 34))

        duration_timestep = self.duration_timestep #in msec
        imgs_per_frame = (int) (duration_timestep * 1e-3 / spike_train.time_scale)
        for i in range(num_timesteps):
            frame_start = imgs_per_frame * i 
            frame_end = frame_start + imgs_per_frame
            mask = (spike_train.ts >= frame_start) & (spike_train.ts < frame_end)
            for x, y, p in zip(spike_train.x[mask], spike_train.y[mask], spike_train.p[mask]):
                frame[
                    i, int(p), int(y), int(x)
                ] = 1
        # print(torch.nn.functional.one_hot(torch.tensor(self._labels[index]), num_classes=10))
        return frame.float(), torch.nn.functional.one_hot(torch.tensor(self._labels[index]), num_classes=10)
