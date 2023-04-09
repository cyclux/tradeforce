"""_summary_
"""

from tradeforce import Tradeforce

CONFIG = {
    "trader": {
        "budget": 10000,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
            "amount_invest_per_asset": 100,
            "investment_cap": 0,
            "buy_performance_score": 0.10,
            "buy_performance_boundary": 0.05,
            "buy_performance_preference": 1,
            "_hold_time_increments": 1000,
            "profit_factor_target": 1.10,
            "profit_factor_target_min": 1.01,
            "moving_window_hours": 180,
        },
    },
    "backend": {
        "dbms": "postgresql",
        "dbms_host": "docker_postgres",
        "dbms_port": 5432,
        "dbms_connect_db": "postgres",
        "dbms_user": "postgres",
        "dbms_pw": "postgres",
        "local_cache": True,
    },
    "market_history": {
        "name": "bitfinex_USD_bfx_history_2y",
        # "name": "bfx_history_docker_test2",
        "exchange": "bitfinex",
        "base_currency": "USD",
        "candle_interval": "5min",
        "history_timeframe_days": 20,
        "update_mode": "none",
        # "force_source": "local_cache",
    },
    "simulation": {
        "_subset_size_increments": 10000,
        "subset_amount": 10,
    },
}

HYPERPARAM_SEARCH = {
    "config": {
        "study_name": "test_study",
        "n_trials": 100,
        "n_jobs": 1,
        "direction": "maximize",
        "storage": "backend",
        "load_if_exists": True,
        "sampler": "RandomSampler",
        # "pruner": "HyperbandPruner",
    },
    "params": {
        "_moving_window_increments": {"min": 20, "max": 220, "step": 20},
        "buy_performance_score": {"min": -0.05, "max": 0.25, "step": 0.05},
        "buy_performance_boundary": {"min": 0.0, "max": 0.15, "step": 0.05},
        "profit_factor_target": {"min": 1.05, "max": 2.5, "step": 0.05},
        "amount_invest_per_asset": {"min": 50, "max": 250, "step": 50},
        "_hold_time_increments": {"min": 1000, "max": 10000, "step": 1000},
        "profit_factor_target_min": {"min": 0.85, "max": 1.1, "step": 0.05},
    },
}


def main():
    sim_result = Tradeforce(CONFIG).run_sim_optuna(HYPERPARAM_SEARCH)
    print(sim_result.best_params)


if __name__ == "__main__":
    main()
