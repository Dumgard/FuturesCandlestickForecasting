import torch
import torch.nn as nn
import torch.nn.functional as F
from cw_rnn import CwRNNCell
from data_preparation import DAIN, RevIN

from typing import Callable, Iterable

STATE_KEEPING_FILE = 'states.pt'


class Model(nn.Module):

    __dict = {
        'Linear': nn.Linear,
        'Dropout': nn.Dropout,
        'AlphaDropout': nn.AlphaDropout,
        'RNNCell': nn.RNNCell,
        'GRUCell': nn.GRUCell,
        'LSTMCell': nn.LSTMCell,
        'CwRNNCell': CwRNNCell,
        'BatchNorm1d': nn.BatchNorm1d,
        'Conv1d': nn.Conv1d,
        'Identity': nn.Identity,
    }
    __lazy_dict = {
        'Linear': nn.LazyLinear,
        'Dropout': nn.Dropout,
        'AlphaDropout': nn.AlphaDropout,
        'RNNCell': nn.RNNCell,
        'GRUCell': nn.GRUCell,
        'LSTMCell': nn.LSTMCell,
        'CwRNNCell': CwRNNCell,
        'BatchNorm1d': nn.LazyBatchNorm1d,
        'Conv1d': nn.LazyConv1d,
        'Identity': nn.Identity,
    }

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 layers: Iterable = None,
                 activation: Callable = torch.relu,
                 activation_after_rnn: Callable = nn.Identity(),
                 dain_params: dict = None,
                 revin_params: dict = None,
                 ask_avg: bool = True,
                 last_layer_activation: bool = False,
                 lazy: bool = False):
        """
        :param input_size:              Number of input features
        :param output_size:             Number of output neurons
        :param layers:                  Layers of model's list with parameters, example:
                                            [('Linear', **kwargs), ('LSTM', **kwargs), ... ]
        :param activation:              Activation function from torch.nn.F
        :param activation_after_rnn:    Activation function that should be applied after RNN layers
        :param dain_params:             Params of DAIN layer. If normalization is not required then set it to None.
        :param revin_params:            Params of RevIN layers. If normalization is not required then set it to None.
                                        Model must contain not more then one of (RevIN, DAIN), not both.
        :param ask_avg:                 Indicates whether the averages are collected externally or
                                            whether DAIN must calculate them itself (case of B_sz > 1)
        :param last_layer_activation:   Indicates whether the activation is used on the last layer
        :param lazy:                    Indicates whether we use lazy modules or not
        """
        assert dain_params is None or revin_params is None

        super(Model, self).__init__()
        __dict = Model.__lazy_dict if lazy else Model.__dict
        self.layers = nn.ModuleList()
        self.activation = activation
        self.activation_after_rnn = activation_after_rnn
        self._lla = last_layer_activation
        self.lazy = lazy
        self.input_size = input_size
        self.output_size = output_size

        self.rnn_states = [None] * len(layers)
        self.NORM = DAIN(input_size=input_size, **dain_params, ask_avg=ask_avg) if dain_params is not None else \
            RevIN(normalization_size=input_size, **revin_params, ask_avg=ask_avg) if revin_params is not None else None
        self.ask_avg = ask_avg
        if dain_params is None and revin_params is None:
            print('*' * 24)
            print('BE SURE TO PROVIDE NORMALIZED DATA SINCE NEITHER DAIN NOR RevIN ARE USED')
            print('*' * 24)

        if layers is None:
            self.layers.append(nn.Linear(input_size, output_size))
            self.last_layer = nn.Identity
        else:
            for layer, kwargs in layers[:-1]:
                self.layers.append(__dict[layer](**kwargs))
            _a, _b = layers[-1]
            assert _a == 'Linear' or _a == 'Identity'
            if _a == 'Identity':
                self._lla = False
            self.last_layer = __dict[_a](**_b)

    def forward(self, x, slice_mean=None, slice_std=None):
        """
        :param x:               x.shape == (B_sz, input_size)
        :param slice_mean:      Mean of time-slice (required for DAIN, provided by Dataset)
        :param slice_std:       Std of time-slice (required for DAIN, provided by Dataset)
        """
        assert (self.ask_avg and slice_mean is not None and slice_std is not None) or not self.ask_avg

        x = x if len(x.shape) > 1 else x.view(1, -1)

        if self.NORM is not None:
            x = self.NORM(x, slice_mean, slice_std) if self.ask_avg else self.NORM(x)

        for i, layer in enumerate(self.layers):

            if isinstance(layer, CwRNNCell):
                # inp.shape = (B_sz, input_size), out.shape = (B_sz, out_size)
                # hx.shape = (B_sz, out_size) == out.shape

                if self.rnn_states[i] is None:
                    # this is needed to keep track of time while still able to save/load states to/from file
                    self.rnn_states[i] = [None, 1]
                else:
                    self.rnn_states[i][0] = self.rnn_states[i][0].detach()

                last_hidden, cur_t = self.rnn_states[i]
                x = layer(x, hx=last_hidden, current_time=cur_t)
                self.rnn_states[i] = [x[-1, :].view(1, -1), layer.curr_t]
                x = self.activation_after_rnn(x)

            elif isinstance(layer, nn.GRUCell) or isinstance(layer, nn.RNNCell):
                # inp.shape = (B_sz, input_size), out.shape = (B_sz, out_size)
                # hx.shape = (B_sz, out_size) == out.shape
                last_hidden = self.rnn_states[i]
                if last_hidden is not None:
                    _b, _bp = x.shape[0], last_hidden.shape[0]
                    last_hidden = last_hidden[_bp - _b:, :].detach()
                x = layer(x, last_hidden)
                self.rnn_states[i] = x
                x = self.activation_after_rnn(x)

            elif isinstance(layer, nn.LSTMCell):
                # inp == x, (h0, c0); x.shape = (B_sz, input_size), h0.shape = c0.shape = (B_sz, hx_size)
                # out == (h1, c1), h1.shape = c1.shape = h0.shape = c0.shape
                last_hidden = self.rnn_states[i]
                if last_hidden is not None:
                    _b, _bp = x.shape[0], last_hidden[0].shape[0]
                    last_hidden[0] = last_hidden[0][_bp - _b:, :].detach()
                    last_hidden[1] = last_hidden[1][_bp - _b:, :].detach()
                x = layer(x, last_hidden)
                self.rnn_states[i] = list(x)  # [x[0][-1, :].view(1, -1), x[1][-1, :].view(1, -1)]
                x = self.activation_after_rnn(x[0])

            elif isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv1d) or \
                    isinstance(layer, nn.LazyLinear) or isinstance(layer, nn.LazyConv1d):
                # Linear:   inp.shape = (*, input_size), out.shape = (*, out_size)
                # Conv:     inp.shape = (B_sz, Ch_in, Len_in), out.shape = (B_sz, Ch_out, Len_out)
                x = self.activation(layer(x))

            elif isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.Dropout) or \
                    isinstance(layer, nn.AlphaDropout) or isinstance(layer, nn.LazyBatchNorm1d):
                # no activation; inp.shape == out.shape; x must be a BATCH
                # for batch norm x must be a batch: (B_sz, input_size) or (B_sz, input_size, L)
                x = layer(x)

        x = self.activation(self.last_layer(x)) if self._lla else self.last_layer(x)

        if self.NORM is not None:
            x = self.NORM(x, slice_mean, slice_std, reverse=True) if self.ask_avg else self.NORM(x, reverse=True)

        return x

    def reset(self):
        """
        Reset current hidden states
        :return:
        """
        self.rnn_states = [None] * len(self.layers)

    def save_state(self):
        """
        Saves hidden-states of the model to file
        :return:
        """
        torch.save(self.rnn_states, STATE_KEEPING_FILE)

    def load_state(self):
        """
        Loads hidden-states of the model from file
        :return:
        """
        self.rnn_states = torch.load(STATE_KEEPING_FILE)
