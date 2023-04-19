from optuna.study import Study

from concurrent import futures
from tradeforce import Tradeforce


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
        "profit_factor_target_min": {"min": 0.85, "max": 1.1, "step": 0.05},
        "amount_invest_per_asset": {"min": 50, "max": 250, "step": 50},
        "hold_time_days": {"min": 1, "max": 100, "step": 1},
    },
}

N_WORKERS = 8


def run_wrapper() -> Study:
    return Tradeforce().run_sim_optuna(HYPERPARAM_SEARCH)


if __name__ == "__main__":
    with futures.ProcessPoolExecutor() as executor:
        pool = [executor.submit(run_wrapper) for worker in range(N_WORKERS)]

        for i in futures.as_completed(pool):
            study_result = i.result()
            print(f"Score (mean profit - std): {study_result.best_value}")
