""" examples/dedicated_market_server.py

Tradeforce can be configured as a dedicated market server. This enables
cluster setups where the database is run on a separate machine providing
access to multiple Tradeforce instances / deployments simultaneously.
For instance, running multiple Optuna hyperparameter search instances
within a clustered environment like Kubernetes.

Relevant configuration options are:
- market_history.update_mode: Needs to be set to "live" which ensures that
                                the market history gets real-time updates
                                via Websocket streams.

- market_history: If no history named 'market_history.name' exists, a new
    history will be fetched from the exchange with the specified settings:
    base_currency, candle_interval, fetch_init_timeframe_days.

- backend.local_cache: Not mandatory to set True. However, it is recommended
    for faster loading times after a restart or usage in simulations.

See README.md for more information about the Tradeforce configuration options.

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
        "local_cache": True,
    },
    "market_history": {
        "name": "bitfinex_history",
        "exchange": "bitfinex",
        "base_currency": "USD",
        "candle_interval": "5min",
        "fetch_init_timeframe_days": 100,
        "update_mode": "live",
    },
}


def main() -> None:
    Tradeforce(CONFIG).run()


if __name__ == "__main__":
    main()
