import torch
from torch import nn


class MSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        diff = input - target
        return torch.mean(torch.pow(diff, 2))


class MAE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        diff = input - target
        return torch.mean(torch.abs(diff))


class VectorSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        diff = input - target
        return torch.pow(diff, 2)


class VectorAE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        diff = input - target
        return torch.abs(diff)



