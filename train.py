import pandas
import torch
import numpy as np
from torch.utils.data import DataLoader
from data_preparation import KlineDataset
from copy import deepcopy
from tqdm import tqdm
from loss import LogCoshLoss, XSigmoidLoss, XTanhLoss, MSELoss, MaskedLoss
from accuracy import MSE
from hf_optim import HessianFree, HFWrapper
from models import Model
from plotting import plot_group
from preprocessing import train_test_split_by_days
import matplotlib.pyplot as plt


def train(model, optimizer, criterion, scorer, dataloaders, n_epochs=10, batch_size=64, use_DAIN=False, HF=False,
          lr_scheduler=None, keep_best=True, verbose=True):

    losses = []
    accuracy = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \tval_acc {v_acc:0.4f}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_train, data_val = dataloaders
    dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=False)
    dataloader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False)

    best_score = 1000000000000000.
    best_model_wts = deepcopy(model.state_dict())

    model = model.to(device)
    model.train()

    with tqdm(desc='epoch', total=n_epochs) as pbar_outer:

        for epoch in range(n_epochs):
            # data_train.reopen_file()
            # data_val.reopen_file()
            if verbose:
                print(f"   Epoch {epoch + 1}")

            # print(len(data_train))

            model.reset()

            train_loss = fit_epoch(model, optimizer, criterion, scorer, dataloader_train, device, lr_scheduler, use_DAIN, HF, verbose=verbose)
            val_acc = eval_epoch(model, criterion, scorer, dataloader_val, device, use_DAIN, verbose=verbose)

            losses.append(train_loss)
            accuracy.append(val_acc)

            if verbose:
                pbar_outer.update(1)
                tqdm.write(log_template.format(ep=epoch + 1, t_loss=sum(train_loss) / len(train_loss),
                                               v_acc=sum(val_acc) / len(val_acc)))

            if keep_best and val_acc[-1] < best_score:
                best_score = val_acc[-1]
                best_model_wts = deepcopy(model.state_dict())

    if keep_best:
        model.load_state_dict(best_model_wts)
    return losses, accuracy


def fit_epoch(model, optimizer, criterion, scorer, data_loader, device, lr_scheduler=None, use_DAIN=False, HF=False, verbose=False):
    losses = []
    model.train()
    counter = 0

    for X, Y, t in data_loader:
        # if counter % 1000 == 999:
        #     print(f"Train batch {counter+1}")
        counter += 1
        Y = Y.to(device).float().requires_grad_()
        if HF:
            optimizer.opt.zero_grad()
            if use_DAIN:     # Should be updated
                X, mean, std = X
                X = X.to(device).float().requires_grad_()
                mean = mean.to(device).float().requires_grad_()
                std = std.to(device).float().requires_grad_()
                loss = optimizer.step(X, Y, mean, std)
            else:
                X = X.to(device).float().requires_grad_()
                loss = optimizer.step(X, Y)
            if verbose:
                print("Train Loss", loss)

        else:
            if use_DAIN:     # Should be updated
                X, mean, std = X
                X = X.to(device).float().requires_grad_()
                mean = mean.to(device).float().requires_grad_()
                std = std.to(device).float().requires_grad_()
                Y_hat = model(X, mean, std)
            else:
                X = X.to(device).float().requires_grad_()
                Y_hat = model(X)

            loss = criterion(Y, Y_hat)
            score = scorer(Y, Y_hat)
            if verbose:
                print("Train Loss, score ", loss, score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #print(X, X_size, total)
        losses.append(loss.item())

    if lr_scheduler is not None:
        lr_scheduler.step()

    return losses


def eval_epoch(model, criterion, scorer, data_loader, device, use_DAIN=False, verbose=False):
    scores = []
    counter = 0
    model.eval()

    with torch.no_grad():
        for X, Y, t in data_loader:
            # if counter % 1000 == 999:
            #     print(f"Val batch {counter+1}")
            counter += 1
            Y = Y.to(device).float()

            if use_DAIN:
                X, mean, std = X
                X = X.to(device).float().requires_grad_()
                mean = mean.to(device).float().requires_grad_()
                std = std.to(device).float().requires_grad_()
                Y_hat = model(X, mean, std)
            else:
                X = X.to(device).float()
                Y_hat = model(X)

            score = scorer(Y, Y_hat)
            if verbose:
                print("Val score ", score)

            scores.append(score.item())

    return scores


if __name__ == "__main__":

    feature_cols = ['o', 'c', 'h', 'l', 'v', 'V', 'n']
    target_cols = feature_cols
    loss_mask = [4, 5, 6]
    time_col = 't'
    filename_train = 'HistoricalPrices\\BTCUSDT_CandlestickHist_train.csv'
    filename_test = 'HistoricalPrices\\BTCUSDT_CandlestickHist_test.csv'

    keep_time = True
    window = 64
    ask_avg = False
    provide_avg = False
    denormalize_output = False
    inp_size = len(feature_cols)

    Data_train = KlineDataset(target_cols=target_cols, feature_cols=feature_cols, provide_avg=provide_avg,
                              denormalize_output=denormalize_output,
                     time_col=time_col, keep_time=keep_time, data_file=filename_train, prediction_window=window)
    Data_val = KlineDataset(target_cols=target_cols, feature_cols=feature_cols, provide_avg=provide_avg,
                            denormalize_output=denormalize_output,
                     time_col=time_col, keep_time=keep_time, data_file=filename_test, prediction_window=window)

    model = Model(inp_size, inp_size, [("CwRNNCell", {"input_size": inp_size, "module_size": 8, "n_modules": 4}),
                         ("Linear", {"in_features": 32, "out_features": 32}),
                         ("Linear", {"in_features": 32, "out_features": inp_size})],
                  revin_params={}, ask_avg=ask_avg)

    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # criterion = MSELoss()
    # criterion = LogCoshLoss()
    # criterion = XSigmoidLoss()
    criterion = XTanhLoss()
    criterion = MaskedLoss(input_size=inp_size, loss=criterion, unused_cols=loss_mask)
    scorer = MSE()

    LR = 0.5
    # optimizer = torch.optim.Adam([
    #     {"params": model.layers.parameters()},
    #     {"params": model.DAIN.mean_layer.parameters(), "lr": LR * model.DAIN.mean_lr},
    #     {"params": model.DAIN.scaling_layer.parameters(), "lr": LR * model.DAIN.scale_lr},
    #     {"params": model.DAIN.gating_layer.parameters(), "lr": LR * model.DAIN.gate_lr},
    # ], lr=LR)

    opt = HessianFree(model.parameters(), lr=LR)
    optimizer = HFWrapper(model, criterion, opt, use_DAIN=False)

    # print(len(Data))
    # G = pandas.read_csv(filename, sep=';')
    # print(G.shape)
    # print('average_window: ', max(window, 8))

    loss, acc = train(model, optimizer, criterion, scorer, (Data_train, Data_val), 2, window, False, True)

    # plt.plot(loss)
    # plt.plot(acc)
    # plt.show()

    plot_group(loss)
    plot_group(acc)




