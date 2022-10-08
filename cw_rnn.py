import torch
import torch.nn as nn

# Thanks ToruOwO for an implementation: https://github.com/ToruOwO/clockwork-rnn-pytorch/blob/master/model.py


class CwRNNCell(nn.Module):
    def __init__(self, input_size, module_size, n_modules, activation=torch.tanh, period_function=None):
        """
        :param input_size:          Number of input features
        :param module_size:         Number of neurons in one hidden module  (k)
        :param n_modules:           Number of hidden modules                (g)
        :param activation:           Activation used for hidden layer
        :param period_function:     Function that is used to define periods of each module:
                period_function(g) == [T_1, T_2, T_3, ..., T_g]
        """
        super(CwRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = n_modules * module_size
        self.rnn_cell = nn.RNNCell(input_size, n_modules * module_size)

        self.activation = activation
        self._g = n_modules
        self._k = module_size
        self.module_period = \
            [2 ** t for t in range(n_modules)] if period_function is None else period_function(self._g)

        self.curr_t = 1

    def step(self, x, hidden, t):
        """
        Only update block-rows that correspond to the executed modules.
        :param x:       Current input -> X_i                (Vector-row, shape == (B_sz, input_size))
        :param hidden:  Last hidden state -> hx_(i-1)       (Vector-row))
        :param t:       Current timestamp (int) to define which modules are used; t >= 1
        """

        update_mask = []
        for T in self.module_period:
            update_mask += [1 if t % T == 0 else 0] * self._k
        update_mask = torch.tensor(update_mask, dtype=torch.int).view(-1, len(update_mask))     # Vector-row
        hidden_out = hidden * (1 - update_mask)     # 1 - update_mask is a retain_mask

        yi = torch.mm(      # mask matrix, then multiply
            x, self.rnn_cell.weight_ih.transpose(1, 0) * update_mask.repeat(self.input_size, 1)
        )   # wi_update_mask == update_mask.repeat(1, self.input_size)
        yi = torch.add(     # add bias only at updated rows
            yi, self.rnn_cell.bias_ih * update_mask
        )

        yh = torch.mm(
            hidden, self.rnn_cell.weight_hh.transpose(1, 0) * update_mask.repeat(update_mask.shape[1], 1)
        )   # wh_update_mask == update_mask.repeat(1, update_mask.shape[0])
        yh = torch.add(
            yh, self.rnn_cell.bias_hh * update_mask
        )

        hidden_out += self.activation(torch.add(yi, yh))
        return hidden_out

    def forward(self, input, hx=None, current_time=None):
        """
        :param input:           shape == (B_sz, input_size)
        :param hx:              shape == (1, hidden_size) == (1, n_modules * module_size)
        :param current_time:        self.curr_t Parameter for Free-Hessian Optimization compatibility
        """
        t, input_size = input.shape
        assert input_size == self.input_size

        if current_time is not None:
            self.curr_t = current_time

        if hx is None:
            hx = torch.zeros(self.hidden_size).view(1, -1)
            self.curr_t = 1

        x_out = []
        for i in range(t):
            y = self.step(input[i, :].view(1, -1), hx, i + self.curr_t)  # keeps track of time at training/running
            hx = y
            x_out.append(y)

        self.curr_t += t

        # output shape (seq_len, output_size) == (B_sz, output_size)
        return torch.stack(x_out, dim=0).view(len(x_out), hx.shape[1])
