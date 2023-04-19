"""_summary_
"""

from tradeforce import Tradeforce

CONFIG = {
    "trader": {
        "id": 3,
        "run_live": True,
        "budget": 0,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
            "amount_invest_per_asset": 1000,
            "investment_cap": 0,
            "buy_performance_score": 0.10,
            "buy_performance_boundary": 0.05,
            "buy_performance_preference": 1,
            "hold_time_days": 4,
            "profit_factor_target": 1.10,
            "profit_factor_target_min": 1.01,
            "moving_window_hours": 160,
        },
    },
    "backend": {
        "dbms": "postgresql",
        "dbms_host": "docker_postgres",
        "dbms_port": 5433,
        "dbms_connect_db": "postgres",
        "dbms_user": "postgres",
        "dbms_pw": "postgres",
        "local_cache": False,
    },
    "market_history": {
        "name": "bfx_history_docker_test",
        "exchange": "bitfinex",
        "base_currency": "USD",
        "candle_interval": "5min",
        "fetch_init_timeframe_days": 60,
        "update_mode": "live",
    },
}


def main() -> None:
    Tradeforce(config=CONFIG).run()


if __name__ == "__main__":
    main()
