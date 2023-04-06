"""_summary_

"""
from __future__ import annotations
from typing import TYPE_CHECKING
import optuna
from tradeforce import simulator
from tradeforce.simulator.utils import to_numba_dict

# from tradeforce.simulator.default_strategies import pre_process
# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce import Tradeforce


def determine_param_type(param_vals):
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


def dict_approach(root: Tradeforce, params: dict, trial: optuna.trial.Trial):
    sim_params = root.config.to_dict()
    for param, val in params.items():
        param_type = determine_param_type(val)
        if param_type == "int":
            sim_params[param] = trial.suggest_int(param, val["min"], val["max"], step=val["step"])
        elif param_type == "float":
            sim_params[param] = trial.suggest_float(param, val["min"], val["max"], step=val["step"])

    sim_params_numba = to_numba_dict(sim_params)
    return sim_params_numba


def direct_approach(root: Tradeforce, params: dict, trial: optuna.trial.Trial):
    for param, val in params.items():
        param_type = determine_param_type(val)
        if param_type == "int":
            setattr(root.config, param, trial.suggest_int(param, val["min"], val["max"], step=val["step"]))
        elif param_type == "float":
            setattr(root.config, param, trial.suggest_float(param, val["min"], val["max"], step=val["step"]))


def run(root: Tradeforce, optuna_config) -> optuna.study.Study:
    config = optuna_config["config"]
    params = optuna_config["params"]

    optuna_db_name = "optuna"
    root.backend.db_exists_or_create(optuna_db_name)

    def objective(trial):
        direct_approach(root, params, trial)
        sim_result = simulator.run(root)
        return sim_result["profit"]

    study = optuna.create_study(
        study_name=config["study_name"],
        direction=config["direction"],
        storage=root.backend.construct_uri(optuna_db_name),
        load_if_exists=config["load_if_exists"],
        sampler=optuna_samplers.get(config.get("sampler", lambda: None), lambda: None)(),
        pruner=optuna_pruners.get(config.get("pruner", lambda: None), lambda: None)(),
    )
    n_jobs = config.get("n_jobs", 1)
    n_trials = config.get("n_trials", 100)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study
