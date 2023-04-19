"""
This example shows how to run a dedicated "market db server" with live candle updates.

Tradeforce can be configured to run as a dedicated market server.
This is useful if you want to run the market DB on a separate machine
and connect multiple traders / simulations to it.
This is especially useful if you want to run Optuna hyperparameter search
in a cluster environment like Kubernetes.
"""

from tradeforce import Tradeforce

CONFIG = {
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
        "name": "bfx_market_server_test2",
        "exchange": "bitfinex",
        "base_currency": "USD",
        "candle_interval": "5min",
        "fetch_init_timeframe_days": 120,
        "update_mode": "live",
    },
}


def main() -> None:
    Tradeforce(CONFIG).run()


if __name__ == "__main__":
    main()
