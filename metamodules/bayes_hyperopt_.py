from typing import Callable
from bayes_opt import BayesianOptimization
import metamodules.black_box_


def bayes_hyperopt(function: Callable,
                   pbounds: dict,
                   init_points: int = 7,
                   n_iter: int = 13,
                   fixed_params: dict = None,
                   minimize: bool = True,
                   random_state: int = 42,
                   ) -> tuple:
    """
    :param function:        Function that is to be optimization
    :param pbounds:         pbounds of BayesianOptimization
    :param init_points:     init_points of BayesianOptimization
    :param n_iter:          n_iter of BayesianOptimization
    :param fixed_params:    Parameters of the function, that must remain fixed
    :param minimize:        If True then target-function will be minimized, else maximized
    :param random_state:    random_state of BayesianOptimization
    :return:                tuple(target value, dict of best non-fixed parameters)
    """
    _black_box = metamodules.black_box_.BlackBox(function=function, fixed_params=fixed_params, minimize=minimize)
    _opt = BayesianOptimization(
        f=_black_box,
        pbounds=pbounds,
        random_state=random_state,
    )

    _opt.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )

    return _opt.max['target'], _opt.max['params']
