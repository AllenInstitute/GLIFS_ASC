import torch
import torch.nn as nn

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    # labels = labels.float()
    # print(labels.shape)
    return nn.CrossEntropyLoss()(input, labels)

def mse_on_average(input, target):
    # print(torch.mean(input, 1).shape)
    # print(target.shape)
    # _, labels = target.max(dim=1)
    return torch.nn.functional.mse_loss(target.float(), torch.mean(input, 1))