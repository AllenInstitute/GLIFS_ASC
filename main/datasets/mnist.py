import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torchvision import datasets, transforms

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_set, self.test_set = mnist_generator(self.hparams.mnist_path, self.hparams.mnist_type)

    def train_dataloader(self):
        return tud.DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return tud.DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return tud.DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    @staticmethod
    def add_mnist_args(parent_parser):
        parser = parent_parser.add_argument_group("mnist")
        parser.add_argument("--mnist_path", type=str)
        parser.add_argument("--mnist_type", type=str, choices=["pixel", "line"])
    

def mnist_label_transform(label):
    return F.one_hot(torch.tensor(label),10)

class ReshapeTransform:
    def __init__(self, task):
        self.task = task
    
    def __call__(self, x):
        return x.view((28 * 28, 1)).float() if self.task == "pixel" else x.view((28, 28)).float()

def mnist_generator(root, task, download=False):
    """
    Creates training and testing dataloader for MNIST

    Arguments
    ---------
    root : str
        filepath to MNIST dataset
    task : str
        either "pixel" for pixel-by-pixel MNIST
        or "line" for line-by-line MNIST
    download: bool
        whether dataset should be downloaded
    """
    # Credit: github.com/locuslab/TCN
    if task not in ["pixel", "line"]:
        raise ValueError("invalid task name provided")
    reshape_transform = ReshapeTransform(task)
    train_set = datasets.MNIST(root=root, train=True, download=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,)),
                                   reshape_transform
                               ]),
                               target_transform=transforms.Compose([
                                    mnist_label_transform])
    )
    test_set = datasets.MNIST(root=root, train=False, download=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,)),
                                  reshape_transform
                              ]),
                               target_transform=transforms.Compose([
                                    mnist_label_transform])
    )

    return train_set, test_set
