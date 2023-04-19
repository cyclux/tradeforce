"""_summary_
"""

from concurrent import futures
from tradeforce import Tradeforce
from optuna.study import Study

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
            "hold_time_days": 4,
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
        "study_name": "test_study_multiprocess",
        "n_trials": 12,
        "n_jobs": 1,
        "direction": "maximize",
        "storage": "backend",
        "load_if_exists": True,
        "sampler": "RandomSampler",
        # "pruner": "HyperbandPruner",
    },
    "params": {
        "moving_window_hours": {"min": 10, "max": 1000, "step": 10},
        "buy_performance_score": {"min": -0.05, "max": 0.25, "step": 0.05},
        "buy_performance_boundary": {"min": 0.0, "max": 0.15, "step": 0.05},
        "profit_factor_target": {"min": 1.05, "max": 2.5, "step": 0.05},
        "amount_invest_per_asset": {"min": 50, "max": 250, "step": 50},
        "hold_time_days": {"min": 1, "max": 100, "step": 1},
        "profit_factor_target_min": {"min": 0.85, "max": 1.1, "step": 0.05},
    },
}
N_WORKERS = 8


def run_wrapper() -> Study:
    return Tradeforce(config=CONFIG).run_sim_optuna(HYPERPARAM_SEARCH)


if __name__ == "__main__":
    with futures.ProcessPoolExecutor() as executor:
        pool = [executor.submit(run_wrapper) for worker in range(N_WORKERS)]

        for i in futures.as_completed(pool):
            print(f"Return Value: {i.result()}")
