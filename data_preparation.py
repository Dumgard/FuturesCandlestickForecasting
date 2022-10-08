import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from typing import Iterable
from queue import Queue

"""
    All data is provided as Time Series
"""


class KlineDataset(Dataset):

    def __init__(self, data: pd.DataFrame = None,
                 target_cols: Iterable[str] = None,
                 feature_cols: Iterable[str] = None,
                 prediction_window: int = 1,
                 data_file: str = None,
                 currency_pair: str = None,
                 keep_time: bool = False,
                 time_col: str or int = None,
                 provide_avg: bool = True,
                 avg_window: int = 8,
                 denormalize_output: bool = True,
                 dim_1d: bool = True,
                 ):
        """
        :param data:                DataFrame, used if data_file is not provided
        :param target_cols:         str(column name)
        :param feature_cols:        str(column name)
        :param prediction_window:   (same as PW) data[i] is X, data[i + PW] is Y
        :param data_file:           file name (path) of data file (str)
        :param currency_pair:       currency pair (str)
        :param keep_time:           Determines whether to provide time (t) as 3rd element of forward(x) return
        :param time_col:            str(name of column used as time)
        :param provide_avg:         Determines whether to provide tuple(X, slice_mean, slice_std) instead of just X
        :param avg_window:          (same as AW) X[i] will be normalized by slice X[i + AW - PW],...,X[i + PW - 1]
        :param denormalize_output:       Whether to denormalize Y_real
        :param dim_1d:              If True then x.shape == (input_size), else (1, input_size)
        """

        assert (keep_time and time_col is not None) or not keep_time
        assert data is not None or data_file is not None

        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.time_col = time_col
        self.currency_pair = currency_pair
        self._pw = prediction_window
        self._aw = max(avg_window, prediction_window)
        self._shift = self._aw - self._pw
        self.filename = data_file
        self.keep_time = keep_time

        self.provide_avg = provide_avg or denormalize_output
        self.denorm_y = denormalize_output
        self.dim = (-1,) if dim_1d else (-1, 1)

        if self.filename is None:
            assert prediction_window < len(data)
            self.target_data = torch.Tensor(data[target_cols].to_numpy())
            self.feature_data = torch.Tensor(data[feature_cols].to_numpy())
            self.time = torch.Tensor(data[time_col].to_numpy()) if keep_time else False
            self.__l = self.feature_data.shape[0]
        else:
            self.__l = sum(1 for _ in open(self.filename)) - 1
            self.count = 0
            self.reopen_file()
            self.currency_pair = self.filename.split('\\')[-1].split('_')[0]

    def reopen_file(self):
        print('REOPEN')
        print('LEN: ', len(self))
        print('COUNT: ', self.count)
        self.file = pd.read_csv(self.filename, header=0, chunksize=1, sep=';', dtype='float64')
        self.count = 0
        self.buffer = Queue()
        self.buffer_residual = Queue()

        # recalculate sums
        self.pw_sum = torch.zeros(len(self.feature_cols)).view(*self.dim)
        self.pw_square_sum = torch.zeros(len(self.feature_cols)).view(*self.dim)
        self.pw_sum_y = torch.zeros(len(self.target_cols)).view(*self.dim)
        self.pw_square_sum_y = torch.zeros(len(self.target_cols)).view(*self.dim)

        for i in range(self._aw):
            cur_ = next(self.file)
            if i < self._shift:
                self.buffer_residual.put(cur_)
            else:
                self.buffer.put(cur_)
            self.pw_sum += torch.Tensor(cur_[self.feature_cols].to_numpy()).view(*self.dim)
            self.pw_square_sum += (torch.Tensor(cur_[self.feature_cols].to_numpy()) ** 2).view(*self.dim)
            self.pw_sum_y += torch.Tensor(cur_[self.target_cols].to_numpy()).view(*self.dim)
            self.pw_square_sum_y += (torch.Tensor(cur_[self.target_cols].to_numpy()) ** 2).view(*self.dim)

    def __len__(self):
        return self.__l - self._aw

    def __getitem__(self, idx):
        """
        :param idx:     ignored if file provided as path, each call returns next line in file
        """
        if self.filename is None:
            idx += self._shift      # appropriate shift for idx so avg starts from index 0
            x = self.feature_data[idx].view(*self.dim)
            y = self.target_data[idx + self._pw].view(*self.dim)

            if self.provide_avg:
                slice_mean, slice_std = self.df_slice_avg(self.feature_data[idx - self._shift:idx + self._pw])
                x = (x, slice_mean, slice_std)

                if self.denorm_y:
                    slice_mean_y, slice_std_y = self.df_slice_avg(self.target_data[idx - self._shift:idx + self._pw])
                    y = (y - slice_mean_y) / slice_std_y

            if self.keep_time:
                t = self.time[idx]
                return x, y, t

            return x, y
        else:
            cur_line = self.buffer.get()
            self.count += 1
            if self.count > len(self):
                self.reopen_file()
                cur_line = self.buffer.get()
            new_line = next(self.file)
            self.buffer.put(new_line)
            x = torch.Tensor(cur_line[self.feature_cols].to_numpy()).view(*self.dim)
            y = torch.Tensor(new_line[self.target_cols].to_numpy()).view(*self.dim)

            if self.provide_avg:
                self.buffer_residual.put(cur_line)
                old_line = self.buffer_residual.get()
                x = self.file_slice_avg(z=x, old_line=old_line, new_line=new_line,
                                        columns=self.feature_cols, mode_x=True)

                if self.denorm_y:
                    y = self.file_slice_avg(z=y, old_line=old_line, new_line=new_line,
                                            columns=self.target_cols, mode_x=False)

            if self.keep_time:
                t = cur_line[self.time_col].iloc[0]
                return x, y, t
            return x, y

    def df_slice_avg(self, df_slice):
        slice_mean = torch.mean(df_slice, dim=0).view(*self.dim)
        slice_std = torch.std(df_slice, dim=0, unbiased=False).view(*self.dim)
        slice_std[slice_std < 1.] = 1.

        return slice_mean, slice_std

    def file_slice_avg(self, z, old_line, new_line, columns, mode_x=True):
        z_new = torch.Tensor(new_line[columns].to_numpy()).view(*self.dim)
        z_prev = torch.Tensor(old_line[columns].to_numpy()).view(*self.dim)

        slice_mean = self.pw_sum / self._aw if mode_x else self.pw_sum_y / self._aw
        slice_std = self.pw_square_sum / self._aw if mode_x else self.pw_square_sum_y / self._aw
        slice_std = slice_std - (slice_mean ** 2)
        slice_std[slice_std < 1.] = 1.
        slice_std = torch.sqrt(slice_std)

        if mode_x:
            self.pw_sum += z_new - z_prev
            self.pw_square_sum += z_new ** 2 - z_prev ** 2
        else:
            self.pw_sum_y += z_new - z_prev
            self.pw_square_sum_y += z_new ** 2 - z_prev ** 2

        if mode_x:
            return z, slice_mean, slice_std
        return (z - slice_mean) / slice_std


class KlineOrderbookDataset(Dataset):

    def __init__(self, data: pd.DataFrame = None,
                 kline_target_cols: Iterable[str] = None,
                 kline_feature_cols: Iterable[str] = None,
                 orderbook_target_cols: Iterable[str] = None,
                 orderbook_feature_cols: Iterable[str] = None,
                 prediction_window: int = 1,
                 data_file: str = None,
                 currency_pair: str = None,
                 keep_time: bool = False,
                 time_col: str or int = None,
                 provide_avg: bool = True,
                 dim_1D: bool = True,
                 ):
        """
        :param data:                    DataFrame, used if data_file is not provided
        :param kline_target_cols:       str(column name)
        :param kline_feature_cols:      str(column name)
        :param orderbook_target_cols:   str(column name)
        :param orderbook_feature_cols:  str(column name)
        :param prediction_window:       (same as PW) data[i] is X, data[i + PW] is Y
        :param data_file:               file name (path) of data file (str)
        :param currency_pair:           currency pair (str)
        :param keep_time:               Determines whether to provide time (t) as 3rd element of forward(x) return
        :param time_col:                str(name of column used as time)
        :param provide_avg:             Determines whether to provide tuple(X, slice_mean, slice_std) instead of just X
        :param dim_1D:              If True then x.shape == (input_size), else (1, input_size)
        """

        assert (keep_time and time_col is not None) or not keep_time
        assert data is not None or data_file is not None

        self.kl_feature_cols = kline_feature_cols
        self.kl_target_cols = kline_target_cols
        self.ob_feature_cols = orderbook_feature_cols
        self.ob_target_cols = orderbook_target_cols
        self.time_col = time_col
        self.currency_pair = currency_pair
        self._pw = prediction_window
        self.filename = data_file
        self.keep_time = keep_time

        self.provide_avg = provide_avg
        self.dim = (-1,) if dim_1D else (-1, 1)

        if self.filename is None:
            assert prediction_window < len(data)
            self.kl_target_data = torch.Tensor(data[self.kl_target_cols].to_numpy())
            self.kl_feature_data = torch.Tensor(data[self.kl_feature_cols].to_numpy())
            self.ob_target_data = torch.Tensor(data[self.ob_target_cols].to_numpy())
            self.ob_feature_data = torch.Tensor(data[self.ob_feature_cols].to_numpy())
            self.time = torch.Tensor(data[time_col].to_numpy()) if keep_time else False
            self.__l = self.kl_feature_data.shape[0]
        else:
            self.file = pd.read_csv(data_file, header=0, chunksize=1, sep=';', dtype='float64')
            self.buffer = Queue()
            self.__l = sum(1 for _ in open(self.filename))
            self.count = self._pw

            self.kl_pw_sum = torch.zeros(len(kline_feature_cols)).view(*self.dim)
            self.kl_pw_square_sum = torch.zeros(len(kline_feature_cols)).view(*self.dim)
            self.ob_pw_sum = torch.zeros(len(orderbook_feature_cols)).view(*self.dim)
            self.ob_pw_square_sum = torch.zeros(len(orderbook_feature_cols)).view(*self.dim)

            for i in range(self._pw):
                cur_ = next(self.file)
                self.buffer.put(cur_)
                self.kl_pw_sum += (torch.Tensor(cur_[self.kl_feature_cols].to_numpy())).view(*self.dim)
                self.kl_pw_square_sum += (torch.Tensor(cur_[self.kl_feature_cols].to_numpy()) ** 2).view(*self.dim)
                self.ob_pw_sum += (torch.Tensor(cur_[self.ob_feature_cols].to_numpy())).view(*self.dim)
                self.ob_pw_square_sum += (torch.Tensor(cur_[self.ob_feature_cols].to_numpy()) ** 2).view(*self.dim)

            self.currency_pair = self.filename.split('\\')[-1].split('_')[0]

    def reopen_file(self):
        self.file = pd.read_csv(self.filename, header=0, chunksize=1, sep=';', dtype='float64')
        self.count = self._pw

        self.buffer = Queue()
        self.kl_pw_sum = torch.zeros(len(self.kline_feature_cols)).view(*self.dim)
        self.kl_pw_square_sum = torch.zeros(len(self.kline_feature_cols)).view(*self.dim)
        self.ob_pw_sum = torch.zeros(len(self.orderbook_feature_cols)).view(*self.dim)
        self.ob_pw_square_sum = torch.zeros(len(self.orderbook_feature_cols)).view(*self.dim)

        for i in range(self._pw):
            cur_ = next(self.file)
            self.buffer.put(cur_)
            self.kl_pw_sum += (torch.Tensor(cur_[self.kl_feature_cols].to_numpy())).view(*self.dim)
            self.kl_pw_square_sum += (torch.Tensor(cur_[self.kl_feature_cols].to_numpy()) ** 2).view(*self.dim)
            self.ob_pw_sum += (torch.Tensor(cur_[self.ob_feature_cols].to_numpy())).view(*self.dim)
            self.ob_pw_square_sum += (torch.Tensor(cur_[self.ob_feature_cols].to_numpy()) ** 2).view(*self.dim)

    def __len__(self):
        return self.__l - self._pw

    def __getitem__(self, idx):
        """
        :param idx:     ignored if file provided as path, each call returns next line in file
        """
        if self.filename is None:
            kl_x = self.kl_feature_data[idx].view(*self.dim)
            kl_y = self.kl_target_data[idx + self._pw].view(*self.dim)
            ob_x = self.ob_feature_data[idx].view(*self.dim)
            ob_y = self.ob_target_data[idx + self._pw].view(*self.dim)

            if self.provide_avg:
                ob_slice_mean = torch.mean(self.ob_feature_data[idx:idx + self._pw], dim=0).view(*self.dim)
                ob_slice_std = torch.std(self.ob_feature_data[idx:idx + self._pw], dim=0, unbiased=False).view(
                    *self.dim)
                ob_slice_std[ob_slice_std < 1.] = 1.
                ob_x = (ob_x, ob_slice_mean, ob_slice_std)

                kl_slice_mean = torch.mean(self.kl_feature_data[idx:idx + self._pw], dim=0).view(*self.dim)
                kl_slice_std = torch.std(self.kl_feature_data[idx:idx + self._pw], dim=0, unbiased=False).view(
                    *self.dim)
                kl_slice_std[kl_slice_std < 1.] = 1.
                kl_x = (kl_x, kl_slice_mean, kl_slice_std)

            if self.keep_time:
                t = self.time[idx]
                return (kl_x, kl_y, t), (ob_x, ob_y, t)
            return (kl_x, kl_y), (ob_x, ob_y)
        else:
            cur = self.buffer.get()
            self.count += 1
            if self.count > len(self):
                self.reopen_file()
            new_line = next(self.file)
            self.buffer.put(new_line)
            kl_x = torch.Tensor(cur[self.kl_feature_cols].to_numpy()).view(*self.dim)
            ob_x = torch.Tensor(cur[self.ob_feature_cols].to_numpy()).view(*self.dim)
            kl_y = torch.Tensor(new_line[self.kl_target_cols].to_numpy()).view(*self.dim)
            ob_y = torch.Tensor(new_line[self.ob_target_cols].to_numpy()).view(*self.dim)

            if self.provide_avg:
                kl_slice_mean = self.kl_pw_sum / self._pw
                kl_slice_std = (self.kl_pw_square_sum / self._pw) - (kl_slice_mean ** 2)
                kl_slice_std[kl_slice_std < 1.] = 1.
                kl_slice_std = torch.sqrt(kl_slice_std)
                ob_slice_mean = self.ob_pw_sum / self._pw
                ob_slice_std = (self.ob_pw_square_sum / self._pw) - (ob_slice_mean ** 2)
                ob_slice_std[ob_slice_std < 1.] = 1.
                ob_slice_std = torch.sqrt(ob_slice_std)

                kl_x_new = torch.Tensor(new_line[self.kl_feature_cols].to_numpy()).view(*self.dim)
                ob_x_new = torch.Tensor(new_line[self.ob_feature_cols].to_numpy()).view(*self.dim)

                self.kl_pw_sum += kl_x_new - kl_x
                self.kl_pw_square_sum += kl_x_new ** 2 - kl_x ** 2
                self.ob_pw_sum += ob_x_new - ob_x
                self.ob_pw_square_sum += ob_x_new ** 2 - ob_x ** 2
                kl_x = (kl_x, kl_slice_mean, kl_slice_std)
                ob_x = (ob_x, ob_slice_mean, ob_slice_std)

            if self.keep_time:
                t = cur[self.time_col].iloc[0]
                return (kl_x, kl_y, t), (ob_x, ob_y, t)
            return (kl_x, kl_y), (ob_x, ob_y)


class DAIN(nn.Module):
    """
    DAIN - stands for Deep Adaptive Input Normalization.
    This is an input data normalization layer that requires learning (and so the grads).

    reference: https://arxiv.org/pdf/1902.07892.pdf
    """

    _eps = 1e-1

    def __init__(self,
                 input_size: int,
                 mean_lr: int = 0.00001,
                 gate_lr: int = 0.001,
                 scale_lr: int = 0.00001,
                 ask_avg: bool = True,
                 mode: str = 'full'
                 ):
        """
        :param input_size:      Number of input features
        :param mean_lr:         It may be necessary for the layer to have its own learning rates for each
        :param gate_lr:         linear module as values of matrices must change rather weakly
        :param scale_lr:
        :param ask_avg:         If True then method forward(...) will require precalculated mean and std of
                                input vector X (X.shape == (B_sz, input_size) where B_sz is len time-series' slice)
                                Else method will calculate mean and std over dim zero (so if X is just vector-row, then
                                mean(X) == X, std(X) == 0, be careful with it)
        :param mode:            Indicates how much of the model is used:
                    avg             -   simple average centering
                    adaptive_avg    -   performs only the first step (adaptive averaging)
                    adaptive_scale  -   perform the first + second step (adaptive averaging + adaptive scaling )
                    full            -   preforms all the steps (averaging, scaling, gating)
        """

        super(DAIN, self).__init__()

        assert mode in (None, 'avg', 'adaptive_avg', 'adaptive_scaling', 'full')
        self.mode = 'full' if mode is None else mode

        self.ask_avg = ask_avg
        self.mean, self.std = None, None

        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_size, input_size, bias=False)
        # Initialize weights as the unit matrix E
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_size, input_size))

        # Parameters for adaptive scaling
        self.scaling_layer = nn.Linear(input_size, input_size, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_size, input_size))

        # Parameters for adaptive gating
        self.gating_layer = nn.Linear(input_size, input_size)

    def forward(self, input, slice_mean=None, slice_std=None, reverse=False):
        """
        :param input:       Vector of shape (B_sz, input_size), where B_sz is len of time-series' slice.
                            If X.shape[0] == 1, then the DAIN must ask for precalculated averages.
        :param slice_mean:  Precalculated mean of time-series' slice (of length W defined in global Model)
                                that contains X
        :param slice_std:   Precalculated std --//--
        :param reverse:     Whether we going backward (denormalization) or straight (normalization)
        """

        assert not (input.shape[0] == 1 and not self.ask_avg)
        assert (self.ask_avg and slice_mean is not None and slice_std is not None) or not self.ask_avg

        if reverse:
            if self.mode == 'adaptive_scaling':
                X = input * self.std
            X = X + self.mean
        else:
            self.mean = slice_mean if self.ask_avg else torch.mean(input, dim=0).view(1, -1)

            if self.mode == 'avg':
                X = input - self.mean
                return X

            self.mean = self.mean_layer(self.mean)
            X = input - self.mean

            if self.mode == 'adaptive_avg':
                return X

            self.std = slice_std if self.ask_avg else torch.std(input, dim=0, unbiased=False).view(1, -1)

            self.std = self.scaling_layer(self.std)
            self.std[self.std <= DAIN._eps] = 1
            X = X / self.std

            if self.mode == 'adaptive_scaling':
                return X

            avg = torch.mean(X, dim=0).view(1, -1)
            gate = torch.sigmoid(self.gating_layer(avg))
            X = X * gate

        return X


class RevIN(nn.Module):
    """
    RevIN - stands for Reversible Instance Normalization.
    Symmetrical normalization-denormalization layers with learnable affine transforms.

    reference: https://openreview.net/pdf?id=cGDAkQo1C0p
    """

    _eps = 1e-5

    def __init__(self,
                 normalization_size: int,
                 # denormalization_size: int,
                 ask_avg: bool = True,
                 affine: bool = True,
                 ):
        """
        :param normalization_size:  Number of normalized features
        # :param denormalization_size:    Number of denormalized features
        :param ask_avg:             If True then method forward(...) will require precalculated mean and std of
                                    input vector X (X.shape == (B_sz, input_size) where B_sz is len time-series' slice)
        :param affine:              Whether to use learnable affine transforms or not
        """

        super(RevIN, self).__init__()

        self.norm_size = normalization_size
        # self.denorm_size = denormalization_size
        self.ask_avg = ask_avg
        self.affine = affine
        self.mean, self.std = None, None
        if self.affine:
            self.aff_weight = nn.Parameter(torch.ones(self.norm_size))
            self.aff_bias = nn.Parameter(torch.zeros(self.norm_size))

    def forward(self, input, slice_mean=None, slice_std=None, reverse=False):
        """
        :param input:       Input (x.shape == (B_sz, input_features) == y.shape == (B_sz, output_features))
        :param slice_mean:  Mean of slice
        :param slice_std:   STD of slice
        :param reverse:     Whether we going backward (denormalization) or straight (normalization)
        """

        assert not (input.shape[0] == 1 and not self.ask_avg)
        assert (self.ask_avg and slice_mean is not None and slice_std is not None) or not self.ask_avg

        if reverse:
            if self.affine:
                x = input - self.aff_bias
                x = x / (self.aff_weight + RevIN._eps ** 2)

            x = x * self.std
            x = x + self.mean
        else:
            self.mean = slice_mean if self.ask_avg else \
                torch.mean(input, dim=0).view(1, -1).detach()
            self.std = slice_std if self.ask_avg else \
                torch.sqrt(torch.var(input, dim=0, unbiased=False) + RevIN._eps).view(1, -1).detach()

            x = input - self.mean
            x = x / self.std

            if self.affine:
                x = x * self.aff_weight
                x = x + self.aff_bias
        return x


if __name__ == '__main__':
    pass
    # S = 'E;o;h;l;c;v;V;q;Q;x;b1_price;b2_price;b3_price;b4_price;b5_price;b6_price;b7_price;b8_price;b9_price;b10_price;b11_price;b12_price;b13_price;b14_price;b15_price;b16_price;b17_price;b18_price;b19_price;b20_price;b1_qty;b2_qty;b3_qty;b4_qty;b5_qty;b6_qty;b7_qty;b8_qty;b9_qty;b10_qty;b11_qty;b12_qty;b13_qty;b14_qty;b15_qty;b16_qty;b17_qty;b18_qty;b19_qty;b20_qty;a1_price;a2_price;a3_price;a4_price;a5_price;a6_price;a7_price;a8_price;a9_price;a10_price;a11_price;a12_price;a13_price;a14_price;a15_price;a16_price;a17_price;a18_price;a19_price;a20_price;a1_qty;a2_qty;a3_qty;a4_qty;a5_qty;a6_qty;a7_qty;a8_qty;a9_qty;a10_qty;a11_qty;a12_qty;a13_qty;a14_qty;a15_qty;a16_qty;a17_qty;a18_qty;a19_qty;a20_qty'
    # S = S.split(';')
    #
    # kl_target_cols = S[1:3]
    # kl_feature_cols = S[3:5]
    # ob_target_cols = S[5:7]
    # ob_feature_cols = S[7:11]
    # kl_time_col = S[0]
    # kl_filename = 'C:\\Users\\Dell\\Downloads\\Telegram Desktop\\BTCUSDT_KlineOrderbookHist_1.csv'
    #
    target_cols = ['Open', 'Close', 'High', 'Low']
    feature_cols = ['Open', 'Close', 'High', 'Low', 'Volume']
    time_col = 'Time'
    # filename = 'data\\BTCBUSD_CandlestickHist.csv'
    filename = 'data\\test_file.csv'

    keep_time = True
    window = 5
    average_window = 8

    A = KlineDataset(target_cols=target_cols, feature_cols=feature_cols,
                     time_col=time_col, keep_time=keep_time, data_file=filename,
                     prediction_window=window, avg_window=average_window)

    tmp = A[0]
    x_, y_, t_ = tmp
    print(x_)
    print(y_)
    print(t_)

    d = torch.randn(17, 6)
    d = pd.DataFrame(d, columns=['Open', 'Close', 'High', 'Low', 'Volume', 'Time'])

    B = KlineDataset(data=d, target_cols=target_cols, feature_cols=feature_cols,
                     time_col=time_col, keep_time=keep_time,
                     prediction_window=window, avg_window=average_window)

    print(d)

    x_, y_, t_ = B[0]
    print(x_)
    print(y_)
    print(t_)

    B_sz = 64
    dl = DataLoader(A, B_sz)
    L = len(A)
    for i in range(3):
        print(f'evaluating epoch {i}')
        for j, res in enumerate(dl):
            # print(res)
            # for i in res[0]:
            #     print(i)
            #     print(i.shape)
            # print(res[1])
            # print(res[1].shape)
            # print(res[2])
            print(res[2].shape)
            print(f'Done: {j * B_sz} out of {L} lines.')
    #
    # test_std = torch.Tensor([107.9259, 104.9190,  99.9200, 110.2724,   2.0134]).view(1, -1)
    # true_std_arr = [
    #     [36060.5, 36060.5, 36060.5, 36060.5, 0.469],
    #     [36042.0, 36042.0, 36042.0, 36042.0, 0.001],
    #     [36042.0, 36042.0, 35856.5, 35856.5, 0.906],
    #     [35829.1, 35843.3, 35824.0, 35843.3, 4.397],
    #     [35834.3, 35848.1, 35807.9, 35815.6, 4.650]
    # ]
    # # [118.8176, 111.1648, 122.9051, 117.6586,   2.2511]
    # true_std = torch.std(torch.Tensor(true_std_arr), dim=0, unbiased=False).view(1, -1)
    # true_std2 = torch.std(torch.Tensor(true_std_arr), dim=0, unbiased=True).view(1, -1)
    # print(test_std)
    # print(true_std)
    # print(true_std2)
    # print(test_std - true_std)
    # print(test_std - true_std2)
    #
    #

    #
    # print(len(A), len(B))

    # A = KlineOrderbookDataset(kline_target_cols=kl_target_cols, kline_feature_cols=kl_feature_cols,
    #                           orderbook_target_cols=ob_target_cols, orderbook_feature_cols=ob_feature_cols,
    #                           time_col=kl_time_col, keep_time=keep_time, data_file=kl_filename,
    #                           prediction_window=window)
    #
    #
    #
    # tmp = A[0]
    # print(tmp)
    # x, y, t = tmp[0]
    # print(x)
    # print(y)
    # print(t)
    #
    # x, y, t = tmp[1]
    # print(x)
    # print(y)
    # print(t)

    # d = torch.randn(17, 6)
    # d = pd.DataFrame(d, columns=['Open', 'Close', 'High', 'Low', 'Volume', 'Time'])
    #
    # B = KlineOrderbookDataset(data=d, target_cols=target_cols, feature_cols=feature_cols,
    #                  time_col=time_col, keep_time=keep_time, prediction_window=window)
    #
    # print(d)
    #
    # print(B[0])
    # x, y, t = B[0]
    # print(x)
    # print(y)
    # print(t)
    #
    # print(len(A), len(B))
