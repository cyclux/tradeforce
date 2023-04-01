"""_summary_
"""

import concurrent.futures
from tradeforce import Tradeforce


config = {
    "trader": {
        "id": 1,
        "run_live": False,
        "budget": 1100,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
            "amount_invest_fiat": 100,
            "investment_cap": 0,
            "buy_opportunity_factor": 0.10,
            "buy_opportunity_boundary": 0.05,
            "prefer_performance": 1,
            "hold_time_limit": 1000,
            "profit_factor": 1.70,
            "profit_ratio_limit": 1.01,
            "moving_window_hours": 180,
        },
    },
    "backend": {
        "dbms": "mongodb",
        "dbms_host": "localhost:1234",
        "local_cache": True,
    },
    "market_history": {
        "name": "bfx_history_2y",
        "exchange": "bitfinex",
        "base_currency": "USD",
        "candle_interval": "5min",
        "history_timeframe": "720days",
        "update_mode": "None",
        "force_source": "local_cache",
    },
    "simulation": {
        "snapshot_size": 100000,
        "snapshot_amount": 10,
    },
}

hyperparam_search = {
    "config": {
        "study_name": "test10",
        "n_trials": 2,
        "n_jobs": 1,
        "direction": "maximize",
        "storage": "JournalStorage",
        "load_if_exists": True,
        "sampler": "RandomSampler",
        "pruner": "HyperbandPruner",
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

N_WORKERS = 8


def run_wrapper():
    return Tradeforce(config=config).run_sim_optuna(hyperparam_search)


if __name__ == "__main__":

    with concurrent.futures.ProcessPoolExecutor() as executor:
        pool = [executor.submit(run_wrapper) for worker in range(N_WORKERS)]

        for i in concurrent.futures.as_completed(pool):
            print(f"Return Value: {i.result()}")
