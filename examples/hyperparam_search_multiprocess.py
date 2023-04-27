""" examples/hyperparam_search_multiprocess.py

An example of how to use the hyperparameter search functionality of Tradeforce
with multiprocessing. The example uses the default buy / sell strategy functions.

Technically, the hyperparameter search could include on any kind of configuration
option Tradeforce offers. However, the primariy use case is to optimize the
trade strategy parameters. The default implementation of the buy / sell strategy
functions introduced following parameters:

    - moving_window_hours: Timeframe for moving window to calulate the price signal scores.
    - buy_signal_score: Price signal score within the moving window.
    - buy_signal_boundary: Range of acceptable 'buy_signal_score's as buy signals.
    - buy_signal_preference: Preference for positive, negative, or closest to buy_signal_score.
    - profit_factor_target: Target profit factor to sell assets. e.g. 1.10 = 10% profit.
    - amount_invest_per_asset: Amount of base_currency to invest per asset.
    - hold_time_days: Number of days to hold an asset before reducing to profit_factor_target_min.
    - profit_factor_target_min: Minimum profit factor to sell assets. e.g. 1.01 = 1% profit.

Tradeforce accepts some basic functionality of Optuna. It is mapped to the 'optuna_config' dict.
Which contains two keys: 'config' and 'search_params'.

The 'config' key defines the Optuna configuration options:

    - study_name: Name of the Optuna study. It identifies the current
                    optimization process.

    - n_trials: Number of trials to run. A trial is a single run of the
                    optimization function.

    - direction: Direction of the optimization. Either 'minimize' or 'maximize'.
                    We want to maximize the profit score.

    - load_if_exists: If True, the study will be loaded from the storage
                        if it exists. Otherwise, a new study will be created.

    - sampler: Sampler to use for the study. Either 'TPESampler',
                'RandomSampler' or 'BruteForceSampler'.

    - pruner: Pruner to use for the study. Either 'HyperbandPruner' or 'MedianPruner'.

The 'search_params' key defines the search space for the optimization:

    For example "moving_window_hours": {"min": 10, "max": 1000, "step": 10}
    searches for the optimal int between 10 and 1000 in steps of 10.

    To search for continous values, use floats:
    For example "profit_factor_target": {"min": 1.05, "max": 2.5, "step": 0.05}

    Note that only float and int values are currently supported because the values
    need to be converted to floats for ease of use in combination
    with Numba JIT compilation.

To utilize the full functionality of Optuna, for example many more samplers and pruners,
you can pass an Optuna instance instead of optuna_config["config"]:
The Study object can be passed as 'optuna_study' to run_sim_optuna().
The search space still needs to be declared in optuna_config["search_params"].

See the Optuna documentation for more information: https://optuna.readthedocs.io/en/stable/

See README.md for more information about the Tradeforce configuration options.
"""

from concurrent import futures
from tradeforce import Tradeforce

import optuna

CONFIG = {
    "trader": {
        "budget": 10000,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
            "amount_invest_per_asset": 100,
            "investment_cap": 0,
            "buy_signal_score": 0.10,
            "buy_signal_boundary": 0.05,
            "buy_signal_preference": 1,
            "hold_time_days": 4,
            "profit_factor_target": 1.10,
            "profit_factor_target_min": 1.01,
            "moving_window_hours": 180,
        },
    },
    "backend": {
        "dbms": "postgresql",
        "dbms_host": "docker_postgres",
        "dbms_port": 5433,
        "dbms_connect_db": "postgres",
        "dbms_user": "postgres",
        "dbms_pw": "postgres",
        "local_cache": True,
        "check_db_sync": False,
    },
    "market_history": {
        "name": "bitfinex_history_2y",
        "exchange": "bitfinex",
        "base_currency": "USD",
        "candle_interval": "5min",
        "fetch_init_timeframe_days": 20,
        "update_mode": "none",
        "force_source": "local_cache",
    },
    "simulation": {
        "subset_size_days": 100,
        "subset_amount": 10,
    },
}

HYPERPARAM_SEARCH = {
    "config": {
        "study_name": "test_study_multiprocess5",
        "n_trials": 111,
        "direction": "maximize",
        "storage": "backend",
        "load_if_exists": True,
        "sampler": "RandomSampler",
        # "pruner": "HyperbandPruner",
    },
    "search_params": {
        "moving_window_hours": {"min": 10, "max": 1000, "step": 10},
        "buy_signal_score": {"min": -0.05, "max": 0.25, "step": 0.05},
        "buy_signal_boundary": {"min": 0.0, "max": 0.15, "step": 0.05},
        "profit_factor_target": {"min": 1.05, "max": 2.5, "step": 0.05},
        "amount_invest_per_asset": {"min": 50, "max": 250, "step": 50},
        "hold_time_days": {"min": 1, "max": 100, "step": 1},
        "profit_factor_target_min": {"min": 0.85, "max": 1.1, "step": 0.05},
    },
}

# Determine number of workers / processes
# Usually, you want to use the number of CPU cores available
N_WORKERS = 8

# Create Tradeforce instance
tradeforce = Tradeforce(config=CONFIG)


def wrapper_run_sim_optuna() -> optuna.Study:
    return tradeforce.run_sim_optuna(HYPERPARAM_SEARCH)


def main() -> None:
    with futures.ProcessPoolExecutor() as executor:
        for _ in range(N_WORKERS):
            executor.submit(wrapper_run_sim_optuna)

    study_name = str(HYPERPARAM_SEARCH["config"]["study_name"])
    # The backend module provides a helper function to construct the storage URI
    storage_uri = tradeforce.backend.construct_uri(db_name="optuna")

    # Load Study from DB (storage) to get the results.
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_uri,
    )

    # For further analysis, you can also load the study in a Jupyter notebook
    # See examples/hyperparam_search_result_analysis.ipynb

    print("Results for study:", study_name)
    print("Best value:", study.best_value)
    print("Best parameters", study.best_params)


if __name__ == "__main__":
    main()
