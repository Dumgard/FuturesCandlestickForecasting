import pandas as pd
from hyperopt import hp, tpe, atpe, rand, STATUS_OK, Trials
import optuna

from metamodules.black_box_ import BlackBox, OptunaBlackBox
from metamodules.bayes_hyperopt_ import bayes_hyperopt
from metamodules.hyperopt_ import ho_hyperopt
from metamodules.optuna_ import optuna_hyperopt

from typing import Callable
from plotting import plot_group
from models import Model
from copy import deepcopy
from train import train
from hf_optim import HessianFree, HFWrapper
from loss import XTanhLoss, XSigmoidLoss, MaskedLoss
from accuracy import MSE, MAE
from data_preparation import KlineDataset
from torch.utils.data import DataLoader
from preprocessing import train_test_split_by_days, column_multiplier
import torch
import pickle


def compare_hf_adam(model: Model,
                    criterion: Callable,
                    accuracy: Callable,
                    dataloaders: list,
                    n_epoch: int,
                    batch_size: int,
                    space_adam: dict,
                    space_hf: dict,
                    n_iter: int = 10,
                    verbose: bool = False,
                    ):
    def objective_adam(adam_dict):
        adam_lr = adam_dict['adam_lr']
        adam_b1 = adam_dict['adam_b1']
        adam_b2 = adam_dict['adam_b2']
        cur_model = deepcopy(model)
        losses, accs = train(model=cur_model,
                             optimizer=torch.optim.Adam(cur_model.parameters(), lr=adam_lr, betas=(adam_b1, adam_b2)),
                             criterion=criterion,
                             scorer=accuracy,
                             dataloaders=dataloaders,
                             n_epochs=n_epoch,
                             batch_size=batch_size,
                             verbose=verbose,
                             keep_best=False,
                             )
        # plot_group(losses)
        return {
            # 'loss': sum(losses[-1]) / len(losses[-1]),
            'loss': sum(accs[-1]) / len(accs[-1]),
            'status': STATUS_OK,
            'losses': losses,
            'accuracies': accs,
        }

    def objective_hf(hf_dict):
        hf_lr = hf_dict['hf_lr']
        cur_model = deepcopy(model)
        losses, accs = train(model=cur_model,
                             optimizer=HFWrapper(cur_model, criterion, HessianFree(cur_model.parameters(), lr=hf_lr),
                                                 use_DAIN=False),
                             criterion=criterion,
                             scorer=accuracy,
                             dataloaders=dataloaders,
                             n_epochs=n_epoch,
                             batch_size=batch_size,
                             verbose=verbose,
                             HF=True,
                             keep_best=False,
                             )
        # plot_group(losses)
        return {
            # 'loss': sum(losses[-1]) / len(losses[-1]),
            'loss': sum(accs[-1]) / len(accs[-1]),
            'status': STATUS_OK,
            'losses': losses,
            'accuracies': accs,
        }

    adam_trials = Trials()
    hf_trials = Trials()

    adam_best = ho_hyperopt(
        function=objective_adam,
        space=space_adam,
        n_iter=n_iter,
        algo=tpe.suggest,
        trials=adam_trials,
    )

    hf_best = ho_hyperopt(
        function=objective_hf,
        space=space_hf,
        n_iter=n_iter,
        algo=tpe.suggest,
        trials=hf_trials,
    )

    return adam_trials, hf_trials, adam_best, hf_best


if __name__ == '__main__':

    # Some important constants
    BATCH_SIZE = 64
    N_EPOCH = 4
    N_ITER = 40
    PW = 10
    ASK_AVG = False
    provide_avg = False
    denormalize_output = False
    keep_time = True
    VERBOSE = False
    COLUMN_MULTIPLIER = True

    # Data preparation\separation
    data_filename = 'data/BTCBUSD_CandlestickHist.csv'

    if COLUMN_MULTIPLIER:
        data_filename = column_multiplier(filename=data_filename,
                                          ignore_time=True,
                                          time_col='Time',
                                          backward=False,
                                          sep=';',
                                          dtype='float64',
                                          )

    filename_train, filename_test = train_test_split_by_days(filename=data_filename,
                                                             train_days=30,
                                                             test_days=5,
                                                             window=PW,
                                                             time_step=(60000, 'ms'),
                                                             to_the_end=True)

    space_adam_ = {
        'adam_lr': hp.loguniform('adam_lr', -16, 0),
        'adam_b1': hp.loguniform('adam_b1', -0.7, 0),
        'adam_b2': hp.loguniform('adam_b2', -0.7, 0),
    }

    space_hf_ = {
        'hf_lr': hp.uniform('hf_lr', 0.2, 1),
    }

    # feature_cols = ['o', 'c', 'h', 'l', 'v', 'V', 'n', 'q', 'Q']
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    target_cols = feature_cols
    # loss_mask = [4, 5, 6, 7, 8]
    loss_mask = [4]
    inp_size = len(feature_cols)
    time_col = 'Time'

    model_ = Model(inp_size, inp_size, [("LSTMCell", {"input_size": inp_size, "hidden_size": 32}),
                                        ("Linear", {"in_features": 32, "out_features": 32}),
                                        ("Linear", {"in_features": 32, "out_features": inp_size})],
                   revin_params={}, ask_avg=ASK_AVG)
    criterion_ = XTanhLoss()
    criterion_ = MaskedLoss(input_size=inp_size, loss=criterion_, unused_cols=loss_mask)
    acc_ = MaskedLoss(input_size=inp_size, loss=MSE(), unused_cols=loss_mask)

    train_df = pd.read_csv(filename_train, header=0, sep=';', dtype='float64')
    test_df = pd.read_csv(filename_test, header=0, sep=';', dtype='float64')

    train_dataset = KlineDataset(target_cols=target_cols, feature_cols=feature_cols, provide_avg=provide_avg,
                                 denormalize_output=denormalize_output,
                                 time_col=time_col, keep_time=keep_time, data=train_df, prediction_window=PW)
    test_dataset = KlineDataset(target_cols=target_cols, feature_cols=feature_cols, provide_avg=provide_avg,
                                denormalize_output=denormalize_output,
                                time_col=time_col, keep_time=keep_time, data=test_df, prediction_window=PW)

    # train_dataset = KlineDataset(target_cols=target_cols, feature_cols=feature_cols, provide_avg=provide_avg,
    #                           denormalize_output=denormalize_output,
    #                  time_col=time_col, keep_time=keep_time, data_file=filename_train, prediction_window=PW)
    # test_dataset = KlineDataset(target_cols=target_cols, feature_cols=feature_cols, provide_avg=provide_avg,
    #                         denormalize_output=denormalize_output,
    #                  time_col=time_col, keep_time=keep_time, data_file=filename_test, prediction_window=PW)

    DATA = [
        train_dataset,
        test_dataset
    ]

    adam_trials, hf_trials, adam_best, hf_best = compare_hf_adam(
        model=model_,
        criterion=criterion_,
        accuracy=acc_,
        dataloaders=DATA,
        n_epoch=N_EPOCH,
        space_adam=space_adam_,
        space_hf=space_hf_,
        batch_size=BATCH_SIZE,
        n_iter=N_ITER,
        verbose=VERBOSE,
    )

    with open('trials/hf_adam_compare/Adam_Trials.obj', 'wb') as adam_trials_file:
        pickle.dump(adam_trials, adam_trials_file)
    with open('trials/hf_adam_compare/HF_Trials.obj', 'wb') as hf_trials_file:
        pickle.dump(hf_trials, hf_trials_file)

    print('ADAM')
    print('Best: ', adam_best)
    print(adam_trials.results)
    print(adam_trials.losses())
    print(adam_trials.statuses())

    print('HF')
    print('Best: ', hf_best)
    print(hf_trials.results)
    print(hf_trials.losses())
    print(hf_trials.statuses())
