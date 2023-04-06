"""_summary_
"""

from tradeforce import Tradeforce

CONFIG = {
    "trader": {
        "budget": 10000,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
            "amount_invest_fiat": 100,
            "investment_cap": 0,
            "buy_opportunity_factor": 0.10,
            "buy_opportunity_boundary": 0.05,
            "prefer_performance": 1,
            "hold_time_limit": 1000,
            "profit_factor": 1.10,
            "profit_ratio_limit": 1.01,
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
        "history_timeframe": "20days",
        "update_mode": "none",
        # "force_source": "local_cache",
    },
    "simulation": {
        "snapshot_size": 10000,
        "snapshot_amount": 10,
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
        "moving_window_increments": {"min": 20, "max": 220, "step": 20},
        "buy_opportunity_factor": {"min": -0.05, "max": 0.25, "step": 0.05},
        "buy_opportunity_boundary": {"min": 0.0, "max": 0.15, "step": 0.05},
        "profit_factor": {"min": 1.05, "max": 2.5, "step": 0.05},
        "amount_invest_fiat": {"min": 50, "max": 250, "step": 50},
        "hold_time_limit": {"min": 1000, "max": 10000, "step": 1000},
        "profit_ratio_limit": {"min": 0.85, "max": 1.1, "step": 0.05},
    },
}


def main():
    sim_result = Tradeforce(CONFIG).run_sim_optuna(HYPERPARAM_SEARCH)
    print(sim_result.best_params)


if __name__ == "__main__":
    main()
