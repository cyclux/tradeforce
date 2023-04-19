"""
A simple example of how to use Tradeforce in simulation mode.
Config loaded from dict.
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
        # "force_source": "backend",
    },
    "simulation": {
        "subset_size_days": 34,
        "subset_amount": 10,
    },
}


def main() -> None:
    sim_result = Tradeforce(config=CONFIG).run_sim()
    print("Score (mean profit - std):", sim_result["score"])


if __name__ == "__main__":
    main()
