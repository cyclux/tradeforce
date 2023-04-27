""" examples/simulator_simple_dict.py

A simple example of how to use Tradeforce in simulation mode.
Config loaded from a dictionary, here defined as CONFIG.

Single simulations without hyperparameter optimization are run with the
run_sim() method of the Tradeforce class. If no pre_process, buy_strategy
or sell_strategy functions are passed to run_sim(), the default
implementations will be applied.

See simulator_custom.py for details about the default strategy implementations
and how to customize them.

See README.md for more information about the Tradeforce configuration options.
"""

from tradeforce import Tradeforce

CONFIG = {
    "trader": {
        "budget": 1000,
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
    },
    "market_history": {
        "name": "bfx_new_test",
        "exchange": "bitfinex",
        "base_currency": "USD",
        "candle_interval": "5min",
        "fetch_init_timeframe_days": 100,
        "update_mode": "none",
        "check_db_sync": False,
    },
    "simulation": {
        "subset_size_days": 30,
        "subset_amount": 10,
        "train_val_split_ratio": 0.8,
    },
}


def main() -> None:
    sim_result = Tradeforce(config=CONFIG).run_sim()

    # Score is calculated by:
    # mean(profit subset) - std(profit subset)
    # See docs for more info about the score calculation.
    print("Score training:", sim_result["score"])
    print("Score validation:", sim_result["score_val"])


if __name__ == "__main__":
    main()
