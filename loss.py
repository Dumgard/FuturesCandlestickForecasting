import torch
from torch import nn
from typing import Iterable


class MaskedLoss(nn.Module):

    def __init__(self, loss, input_size, unused_cols=Iterable[int]):
        """
        :param loss:            Any cost-function
        :param unused_cols:     Iterable of numbers of columns that are unused
        """
        super().__init__()
        self.loss = loss
        self.unused = sorted(unused_cols)
        self.mask = torch.ones(input_size)
        self.mask[self.unused] = 0.
        self.mask = self.mask.view(1, -1)

    def forward(self, input, target):
        mask = self.mask.repeat(input.shape[0], 1)
        input = input * mask
        target = target * mask
        return self.loss(input, target)


class LogCoshLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        diff = input - target
        diff[diff < -20.] = -20.
        return torch.mean(diff * torch.log((1 + torch.exp(-2 * diff)) / 2))


class XTanhLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        diff = input - target
        return torch.mean(diff * torch.tanh(diff))


class XSigmoidLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        diff = input - target
        # return torch.mean(diff * (2 / (1 + torch.exp(-diff)) - 1))
        return torch.mean(diff * (2 * torch.sigmoid(diff) - 1))


class MSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        diff = input - target
        return torch.mean(torch.pow(diff, 2))


class MAELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        diff = input - target
        return torch.mean(torch.abs(diff))
