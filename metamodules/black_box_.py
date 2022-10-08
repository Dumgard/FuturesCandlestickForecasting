from typing import Callable
from optuna.integration import TorchDistributedTrial


class BlackBox:

    def __init__(self,
                 function: Callable,
                 fixed_params: dict = None,
                 minimize: bool = True,
                 ):
        """
        Wraps the function to fix some parameters that should not be optimized
        :param function:            Function, that requires optimization
        :param fixed_params:        Parameters of the function, that must remain fixed
        :param minimize:        If True then target-function will be minimized, else maximized
        """
        self.f = function
        self.fixed = dict() if fixed_params is None else fixed_params
        self.min_ = minimize

    def __call__(self, *args, **kwargs):
        return self.f(*args, **self.fixed, **kwargs)    # * (-1 if self.min_ else 1)


class OptunaBlackBox:

    def __init__(self,
                 objective: Callable,
                 fixed_params: dict = None,
                 pruning: bool = True,
                 device=None,
                 ):
        """
        Wraps the function to fix some parameters that should not be optimized
        :param objective:           Function, that requires optimization
        :param fixed_params:        Parameters of the function, that must remain fixed
        :param pruning:             If True then Trial should be integrated for pruning activation
        :param device:              PyTorch device for integrated Trials
        """
        self.f = objective
        self.fixed = dict() if fixed_params is None else fixed_params
        self.pruning = pruning
        self.device = device

    def __call__(self, trial):
        if self.pruning:
            trial = TorchDistributedTrial(trial, device=self.device)
        return self.f(trial, **self.fixed)
