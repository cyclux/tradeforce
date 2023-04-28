""" examples/hyperparam_search.py

An example of how to use the hyperparameter search functionality of Tradeforce.

Technically, the hyperparameter search could include on any kind of configuration
option Tradeforce offers. However, the primariy use case is to optimize the
trade strategy parameters. The example uses the default buy / sell strategy
functions which introduced following parameters:

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

from tradeforce import Tradeforce

CONFIG = {
    "trader": {
        "budget": 1000,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
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
        "name": "bitfinex_history",
        "exchange": "bitfinex",
        "base_currency": "USD",
        "candle_interval": "5min",
        "fetch_init_timeframe_days": 100,
        "update_mode": "none",
        "force_source": "local_cache",
    },
    "simulation": {
        "subset_size_days": 30,
        "subset_amount": 10,
        "train_val_split_ratio": 0.8,
    },
}

HYPERPARAM_SEARCH = {
    "config": {
        "study_name": "test_study",
        "n_trials": 100,
        "direction": "maximize",
        "load_if_exists": True,
        "sampler": "TPESampler",
        "pruner": "HyperbandPruner",
    },
    "search_params": {
        "amount_invest_per_asset": {"min": 50, "max": 250, "step": 50},
        "moving_window_hours": {"min": 10, "max": 1000, "step": 10},
        "buy_signal_score": {"min": -0.05, "max": 0.25, "step": 0.05},
        "buy_signal_boundary": {"min": 0.0, "max": 0.15, "step": 0.05},
        "buy_signal_preference": {"min": -1, "max": 1, "step": 1},
        "profit_factor_target": {"min": 1.05, "max": 2.5, "step": 0.05},
        "hold_time_days": {"min": 1, "max": 100, "step": 1},
        "profit_factor_target_min": {"min": 0.85, "max": 1.1, "step": 0.05},
    },
}


def main() -> None:
    study = Tradeforce(CONFIG).run_sim_optuna(optuna_config=HYPERPARAM_SEARCH)

    print("Results for study:", study.study_name)
    print("Best value:", study.best_value)
    print("Best parameters", study.best_params)


if __name__ == "__main__":
    main()
