"""A simple example of how to use Tradeforce in simulation mode.

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
        "local_cache": False,
    },
    "market_history": {
        "name": "bfx_history_docker_test2",
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


def main():
    sim_result = Tradeforce(config=CONFIG).run_sim()
    print(sim_result)


if __name__ == "__main__":
    main()
