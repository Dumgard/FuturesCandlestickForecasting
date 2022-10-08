from typing import Callable, Optional
import optuna
import metamodules.black_box_
"""
Short example:
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return (x - 2) ** 2
    
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    found_x = best_params["x"]
    
    
    **Trial**: A single call of the objective function
    **Study**: An optimization session, which is a set of trials
    **Parameter**: A variable whose value is to be optimized
    
    # To get the dictionary of parameter name and parameter values:
    study.best_params
    
    # To get the best observed value of the objective function:
    study.best_value
    
    # To get the best trial:
    study.best_trial
    
    # To get all trials:
    study.trials
    
    # By executing :func:`~optuna.study.Study.optimize` again, we can continue the optimization.
    study.optimize(objective, n_trials=100)

Search space:
    For hyperparameter sampling, Optuna provides the following features:
    - :func:`optuna.trial.Trial.suggest_categorical` for categorical parameters
    - :func:`optuna.trial.Trial.suggest_int` for integer parameters
    - :func:`optuna.trial.Trial.suggest_float` for floating point parameters
    
    With optional arguments of ``step`` and ``log``, we can discretize or take the logarithm of 
    integer and floating point parameters.
    
    Examples:
        # Categorical parameter
        optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])
    
        # Integer parameter
        num_layers = trial.suggest_int("num_layers", 1, 3)
    
        # Integer parameter (log)
        num_channels = trial.suggest_int("num_channels", 32, 512, log=True)
    
        # Integer parameter (discretized)
        num_units = trial.suggest_int("num_units", 10, 100, step=5)
    
        # Floating point parameter
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)
    
        # Floating point parameter (log)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    
        # Floating point parameter (discretized)
        drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 1.0, step=0.1)
        
    Example of few model architecture's optimization:
        def objective(trial):
            classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
            if classifier_name == "SVC":
                svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
                classifier_obj = sklearn.svm.SVC(C=svc_c)
            else:
                rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
                classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)
    
    And an example of run-defined params optimization:
        def create_model(trial, in_size):
            n_layers = trial.suggest_int("n_layers", 1, 3)
    
            layers = []
            for i in range(n_layers):
                n_units = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
                layers.append(nn.Linear(in_size, n_units))
                layers.append(nn.ReLU())
                in_size = n_units
            layers.append(nn.Linear(in_size, 10))
    
            return nn.Sequential(*layers)

Sampling Algorithms:
    - Grid Search implemented in :class:`~optuna.samplers.GridSampler`
    - Random Search implemented in :class:`~optuna.samplers.RandomSampler`
    - Tree-structured Parzen Estimator algorithm implemented in :class:`~optuna.samplers.TPESampler`
    - CMA-ES based algorithm implemented in :class:`~optuna.samplers.CmaEsSampler`
    - Algorithm to enable partial fixed parameters implemented in :class:`~optuna.samplers.PartialFixedSampler`
    - Nondominated Sorting Genetic Algorithm II implemented in :class:`~optuna.samplers.NSGAIISampler`
    - A Quasi Monte Carlo sampling algorithm implemented in :class:`~optuna.samplers.QMCSampler`
    The default sampler is :class:`~optuna.samplers.TPESampler`.
    
    Examples:
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
        study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())
        
Pruning Algorithms:
    ``Pruners`` automatically stop unpromising trials at the early stages of the training 
    (a.k.a., automated early-stopping).
    
    - Median pruning algorithm implemented in :class:`~optuna.pruners.MedianPruner`
    - Non-pruning algorithm implemented in :class:`~optuna.pruners.NopPruner`
    - Algorithm to operate pruner with tolerance implemented in :class:`~optuna.pruners.PatientPruner`
    - Algorithm to prune specified percentile of trials implemented in :class:`~optuna.pruners.PercentilePruner`
    - Asynchronous Successive Halving algorithm implemented in :class:`~optuna.pruners.SuccessiveHalvingPruner`
    - Hyperband algorithm implemented in :class:`~optuna.pruners.HyperbandPruner`
    - Threshold pruning algorithm implemented in :class:`~optuna.pruners.ThresholdPruner`

    We use :class:`~optuna.pruners.MedianPruner` in most examples,
    though basically it is outperformed by :class:`~optuna.pruners.SuccessiveHalvingPruner` and
    :class:`~optuna.pruners.HyperbandPruner` as in 
    this benchmark result <https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako>
    
    To turn on the pruning feature, you need to call :func:`~optuna.trial.Trial.report` and 
    :func:`~optuna.trial.Trial.should_prune` after each step of the iterative training.
    :func:`~optuna.trial.Trial.report` periodically monitors the intermediate objective values.
    :func:`~optuna.trial.Trial.should_prune` decides termination of the trial that does not meet a predefined condition.
    
    Example:
        def objective(trial):
        iris = sklearn.datasets.load_iris()
        classes = list(set(iris.target))
        train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
            iris.data, iris.target, test_size=0.25, random_state=0
        )
    
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        clf = sklearn.linear_model.SGDClassifier(alpha=alpha)
    
        for step in range(100):
            clf.partial_fit(train_x, train_y, classes=classes)
    
            # Report intermediate objective value.
            intermediate_value = 1.0 - clf.score(valid_x, valid_y)
            trial.report(intermediate_value, step)
    
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()
    
        return 1.0 - clf.score(valid_x, valid_y)
    
    Integration Modules for Pruning:
        https://optuna.readthedocs.io/en/stable/reference/integration.html#module-optuna.integration
        
        optuna.integration.PyTorchIgnitePruningHandler  -  PyTorch Ignite handler to prune unpromising trials.
        optuna.integration.PyTorchLightningPruningCallback  -  PyTorch Lightning callback to prune unpromising trials.
        optuna.integration.TorchDistributedTrial  -  A wrapper of Trial to incorporate Optuna with PyTorch distributed.
        
        Example:
            def objective(single_trial):
                trial = optuna.integration.TorchDistributedTrial(single_trial)
                ...
            ...
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
Visualisation:
    Optuna provides various visualization features in :mod:`optuna.visualization` to 
    analyze optimization results visually.
    
    Modules:
        You can use Matplotlib instead of Plotly for visualization by simply replacing `optuna.visualization` with
        `optuna.visualization.matplotlib` in the following examples.
        
        from optuna.visualization import plot_contour
        from optuna.visualization import plot_edf
        from optuna.visualization import plot_intermediate_values
        from optuna.visualization import plot_optimization_history
        from optuna.visualization import plot_parallel_coordinate
        from optuna.visualization import plot_param_importances
        from optuna.visualization import plot_slice
        
    
    Plot functions:

        # Visualize the optimization history. See :func:`~optuna.visualization.plot_optimization_history` for the details.
        plot_optimization_history(study)
    
        # Visualize the learning curves of the trials. See :func:`~optuna.visualization.plot_intermediate_values` for the details.
        plot_intermediate_values(study)
    
        # Visualize high-dimensional parameter relationships. See :func:`~optuna.visualization.plot_parallel_coordinate` for the details.
        plot_parallel_coordinate(study)
    
        # Select parameters to visualize.
        plot_parallel_coordinate(study, params=["bagging_freq", "bagging_fraction"])
    
        # Visualize hyperparameter relationships. See :func:`~optuna.visualization.plot_contour` for the details.
        plot_contour(study)
    
        # Select parameters to visualize.
        plot_contour(study, params=["bagging_freq", "bagging_fraction"])
    
        # Visualize individual hyperparameters as slice plot. See :func:`~optuna.visualization.plot_slice` for the details.
        plot_slice(study)
    
        # Select parameters to visualize.
        plot_slice(study, params=["bagging_freq", "bagging_fraction"])
    
        # Visualize parameter importances. See :func:`~optuna.visualization.plot_param_importances` for the details.
        plot_param_importances(study)
    
        # Learn which hyperparameters are affecting the trial duration with hyperparameter importance.
        optuna.visualization.plot_param_importances(
            study, target=lambda t: t.duration.total_seconds(), target_name="duration"
        )
    
        # Visualize empirical distribution function. See :func:`~optuna.visualization.plot_edf` for the details.
        plot_edf(study)

References:
    https://optuna.readthedocs.io/en/stable/index.html
    
"""


def optuna_hyperopt(function: Callable,
                    n_iter: int = 10,
                    sampler: Optional[optuna.samplers.BaseSampler] = optuna.samplers.TPESampler,
                    pruner: Optional[optuna.pruners.BasePruner] = optuna.pruners.MedianPruner,
                    fixed_params: dict = None,
                    minimize: bool = True,
                    device=None,
                    ) -> tuple:
    """
    :param function:        Target function to hyper-optimize. The only non-fixed parameter is optuna's 'trial',
                            others must remain in fixed_params
    :param n_iter:          Max number of iterations
    :param sampler:         Sampler algorithm
    :param pruner:          Pruner algorithm
    :param fixed_params:    Parameters of the function, that must remain fixed
    :param minimize:        If True then target-function will be minimized, else maximized
    :param device:          PyTorch device for integrated Trials
    :return:                tuple(target value, dict of best non-fixed parameters, optuna.Study instance)
    """
    _black_box = metamodules.black_box_.OptunaBlackBox(objective=function, fixed_params=fixed_params, device=device)
    study = optuna.create_study(
        direction="minimize" if minimize else "maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(_black_box, n_trials=n_iter)

    return study.best_value, study.best_params, study
