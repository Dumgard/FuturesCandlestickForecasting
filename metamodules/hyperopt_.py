from typing import Callable
from hyperopt import tpe, atpe, hp, fmin, rand
import metamodules.black_box_


"""
Some hints
    Hyperopt search spaces:
    
        hp.choice(label, list)
            Returns one of the options, which should be a list or tuple.
        hp.randint(label, upper) or hp.randint(label, low, high)
            Returns a random integer in the range [0, upper).
        hp.uniform(label, low, high)
            Returns a value uniformly between low and high.
        hp.quniform(label, low, high, q)
            Returns a value like round(uniform(low, high) / q) * q
        hp.loguniform(label, low, high)
            Returns a value drawn according to exp(uniform(low, high)) 
            so that the logarithm of the return value is uniformly distributed.
        hp.qloguniform(label, low, high, q)
            Returns a value like round(exp(uniform(low, high)) / q) * q
        hp.normal(label, mu, sigma)
            Returns a real value that's normally-distributed with mean mu and standard deviation sigma. 
            When optimizing, this is an unconstrained variable.
        hp.qnormal(label, mu, sigma, q)
            Returns a value like round(normal(mu, sigma) / q) * q
        hp.lognormal(label, mu, sigma)
            Returns a value drawn according to exp(normal(mu, sigma)) 
            so that the logarithm of the return value is normally distributed. 
        hp.qlognormal(label, mu, sigma, q)
            Returns a value like round(exp(normal(mu, sigma)) / q) * q
        hp.pchoice(label, p_list) with p_list as a list of (probability, option) pairs
        hp.uniformint(label, low, high, q) or hp.uniformint(label, low, high) since q = 1.0
        
        :label: is an str representation of parameter's name, so if we have func(x, input2), then 'x' and 'input2'
        will be the labels
        
        For multiple parameter optimization :space: should be a dict. Example:
            space = {
                'x': 3 + hp.loguniform('x', 0, 8),
                'input2': hp.choice('input2', ['yes', 'no', 'maybe', 'i_dont_know']),
            }
        
    Optimization algorithms:
        TPE (Tree of Parzen Estimators)     hyperopt.tpe.suggest
        Random Search                       hyperopt.rand.suggest
        Adaptive TPE                        hyperopt.atpe.suggest
    
    Attaching extra information (Trials):
        return value of the function can be represented as dict:
        return {
            'loss': real_loss_of_target_function,
            'status': hyperopt.STATUS_OK,   # provides info for hyperopt that function is calculated without exceptions
            # and absolutely ANYTHING else you can imagine, examples:
            'anything_else': something,
            'lets_time_our_func': time.time(),
            'accuracy': accuracy,
        } 
        
        And, then the Trials:
        trials = hyperopt.Trials()
        best = fmin(
            objective,
            space=hp.uniform('x', -10, 10),
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)
        )
        
        trials.trials       # a list of dictionaries representing everything about the search
        trials.results      # a list of dictionaries returned by 'objective' during the search
        trials.losses()     # a list of losses (float for each 'ok' trial)
        trials.statuses()   # a list of status strings
        
References:
    http://hyperopt.github.io/hyperopt
    http://proceedings.mlr.press/v28/bergstra13.pdf
    Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in 
    Hundreds of Dimensions for Vision Architectures. TProc. of the 30th International Conference on Machine Learning 
    (ICML 2013), June 2013, pp. I-115 to I-23.
"""


def ho_hyperopt(function: Callable,
                space: dict,
                n_iter: int = 10,
                algo: Callable = tpe.suggest,
                fixed_params: dict = None,
                minimize: bool = True,
                trials=None,
                ) -> dict:
    """
    :param function:        Target function to hyper-optimize
    :param space:           Similar to pbounds (see Hints at the source module - hyperopt_)
    :param n_iter:          Max number of iterations
    :param algo:            Optimization algorithm (one of 3 - tpe.suggest, atpe.suggest, rand.suggest)
    :param fixed_params:    Parameters of the function, that must remain fixed
    :param minimize:        If True then target-function will be minimized, else maximized
    :param trials:          hyperopt.Trials instance that collects info from optimization process (see Hints)
    :return:                dict('name_of_parameter': float(optimized_value))
    """
    _black_box = metamodules.black_box_.BlackBox(function=function, fixed_params=fixed_params, minimize=not minimize)
    _best = fmin(
        fn=_black_box,
        space=space,
        algo=algo,
        max_evals=n_iter,
    ) if trials is None else fmin(
        fn=_black_box,
        space=space,
        algo=algo,
        max_evals=n_iter,
        trials=trials,
    )

    return _best
