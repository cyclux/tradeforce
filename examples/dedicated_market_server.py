from tradeforce import Tradeforce

CONFIG = {
    "backend": {
        "dbms": "postgresql",  # "none, mongodb or postgresql"
        "dbms_host": "docker_postgres",  # docker_postgres, docker_mongodb
        "dbms_port": 5432,
        "dbms_connect_db": "postgres",
        "dbms_user": "postgres",
        "dbms_pw": "postgres",
        "local_cache": True,
    },
    "market_history": {
        "name": "bfx_history_docker_test",
        "exchange": "bitfinex",
        "base_currency": "USD",
        "candle_interval": "5min",
        "history_timeframe": "60days",
        "update_mode": "live",  # none, once or live
    },
}

# To provide a subset, TradingEngine can also receive the optional argument "assets".
# This has to be a list of asset symbols.
# For example: Tradeforce(assets = ["BTC", "ETH", "XRP"]).run()


def main():
    Tradeforce(CONFIG).run_new()


if __name__ == "__main__":
    main()
