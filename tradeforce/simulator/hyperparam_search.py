""" simulator/hyperparam_search.py

Module: tradeforce.simulator.hyperparam_search
----------------------------------------------

Provides functionality for running hyperparameter optimization using Optuna
in conjunction with the Tradeforce trading simulation framework.
The hyperparameter search is performed for a given Tradeforce instance
and Optuna configuration, which specifies the settings for the optimization
"Study" and search parameters for each parameter to be optimized. A "study" is
an Optuna object that contains the results of the optimization process.

The module supports setting the search parameters for an optimization trial,
determining the data type of parameter values, and defining the objective
function for the Optuna study. It also provides a utility function to run
the hyperparameter optimization process and returns an Optuna study object
containing the results.

Main Function:

    run(optuna_config): Run a hyperparameter optimization using Optuna for
                            the given Tradeforce instance and Optuna config,
                            returning an Optuna study object containing the
                            results of the optimization process.

"""
from __future__ import annotations
from typing import TYPE_CHECKING

import optuna
from optuna import Trial
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from tradeforce import simulator

# from tradeforce.simulator.default_strategies import pre_process
# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce import Tradeforce


def determine_param_type(param_vals: dict[str, int | float]) -> str:
    """Determine the data type of the parameter values in the provided dictionary.

    Params:
        param_vals: Dict containing the parameter names and their
                        corresponding values.

    Returns:
        A string indicating the data type ("int" or "float") of the parameter
        values in the provided dictionary. If at least one value is a float,
        the function returns "float"; otherwise, it returns "int".
    """

    is_float = any(isinstance(val, float) for val in param_vals.values())
    param_type = "float" if is_float else "int"
    return param_type


optuna_samplers = {
    "RandomSampler": optuna.samplers.RandomSampler,
    "TPESampler": optuna.samplers.TPESampler,
    "BruteForceSampler": optuna.samplers.BruteForceSampler,
}

optuna_pruners = {
    "HyperbandPruner": optuna.pruners.HyperbandPruner,  # type: ignore
    "MedianPruner": optuna.pruners.MedianPruner,  # type: ignore
}


def _set_search_params(root: Tradeforce, params: dict, trial: optuna.trial.Trial) -> None:
    """Set the search parameters for an optimization trial

    using the given params dictionary. The function updates the config of the provided
    Tradeforce instance with the Optuna suggested parameter values for the current trial.

    For each parameter in the params dictionary, the function determines the data type and
    uses the appropriate suggest_int or suggest_float method from the trial object to
    update the parameter value in the Tradeforce instance's config.

    Params:
        root:    A Tradeforce instance.

        params:  Dict containing the parameter names and their corresponding search
                 space (min, max, step) as nested dictionaries.

        trial:   An Optuna trial object.
    """
    for param, val in params.items():
        param_type = determine_param_type(val)
        if param_type == "int":
            setattr(root.config, param, trial.suggest_int(param, val["min"], val["max"], step=val["step"]))
        elif param_type == "float":
            setattr(root.config, param, trial.suggest_float(param, val["min"], val["max"], step=val["step"]))


def _create_study(root: Tradeforce, config: dict, optuna_db_name: str = "optuna") -> optuna.study.Study:
    # Creates an Optuna study with the specified settings, samplers, and pruners.
    return optuna.create_study(
        study_name=config["study_name"],
        direction=config["direction"],
        storage=root.backend.construct_uri(optuna_db_name),
        load_if_exists=config["load_if_exists"],
        sampler=optuna_samplers.get(config.get("sampler", lambda: None), lambda: None)(),
        pruner=optuna_pruners.get(config.get("pruner", lambda: None), lambda: None)(),
    )


def run(
    root: Tradeforce, optuna_config: dict[str, dict], optuna_study: optuna.study.Study | None
) -> optuna.study.Study:
    """Run a hyperparameter optimization using Optuna

    for the given Tradeforce instance and Optuna configuration dictionary.

    Params:
        root:          A Tradeforce instance.
        optuna_config: Dict containing the configuration settings for the
                        optimization study and search parameters for each parameter
                        to be optimized.

    Returns:
        An Optuna study object containing the results of the optimization process.
    """
    if not hasattr(root.backend, "db_exists_or_create"):
        raise ValueError(
            "Optuna only works with Postgres backend. " + "Please set backend.dbms to 'postgresql' in config file."
        )

    optuna_db_name = "optuna"

    config = optuna_config.get("config", {})
    search_params = optuna_config.get("search_params", {})

    if not search_params:
        raise SystemExit(
            "Optuna search parameters not specified. Needs to be a dict within optuna_config['search_params']"
        )

    root.backend.db_exists_or_create(optuna_db_name)

    # Define objective function for optuna "study"
    def objective(trial: Trial) -> int:
        _set_search_params(root, search_params, trial)
        sim_result = simulator.run_train_val_split(root)
        return int(sim_result["score"])

    study = optuna_study if optuna_study is not None else _create_study(root, config, optuna_db_name)

    n_jobs = config.get("n_jobs", 1)
    n_trials = config.get("n_trials", 100)

    # Runs the optimization process / study
    study.optimize(
        objective,
        n_jobs=n_jobs,
        callbacks=[MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE, TrialState.RUNNING, TrialState.PRUNED))],
        show_progress_bar=True,
    )

    return study
