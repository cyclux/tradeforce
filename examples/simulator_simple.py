"""A simple example of how to use Tradeforce in simulation mode.

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
        "local_cache": False,
    },
    "market_history": {
        "name": "bfx_history_docker_test2",
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


def main():
    sim_result = Tradeforce(config=CONFIG).run_sim()
    print(sim_result)


if __name__ == "__main__":
    main()
