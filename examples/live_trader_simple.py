""" examples/live_trader_simple.py

This is a simple example of a live trader configuration.

Relevant configuration options are:

- trader.run_live:            If True, the trader will run in live mode.

- trader.id:                  The id of the trader. Used to identify the trader.
                                Useful if you want to run multiple traders on the same DB.

- trader.budget:              In live mode, the setting of a budget can be ignored as it is
                                automatically set to the current balance of the
                                base_currency on the exchange.

- trader.maker_fee:           This setting is irrelevant in live mode as the fee is
                                determined by the exchange.

- trader.taker_fee:           This setting is irrelevant in live mode as the fee is
                                determined by the exchange.

- trader.strategy:            The strategy configuration is the same as in simulation mode.

- market_history.update_mode: Needs to be set to "live" which ensures that
                                the market history gets real-time updates
                                via Websocket streams.

See README.md for more information about the Tradeforce configuration options.

DISCLAIMER
----------------------------------------------------------------------------
Use at your own risk! Tradeforce is currently in beta, and bugs may occur.
Furthermore, there is no guarantee that strategies that have performed well
in the past will continue to do so in the future.
----------------------------------------------------------------------------

"""

from tradeforce import Tradeforce

CONFIG = {
    "trader": {
        "id": 1,
        "run_live": True,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
            "amount_invest_per_asset": 100,
            "moving_window_hours": 180,
            "buy_signal_score": 0.10,
            "buy_signal_boundary": 0.05,
            "buy_signal_preference": 1,
            "profit_factor_target": 1.10,
            "hold_time_days": 4,
            "profit_factor_target_min": 1.01,
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
        "name": "bitfinex_history",
        "exchange": "bitfinex",
        "base_currency": "USD",
        "candle_interval": "5min",
        "fetch_init_timeframe_days": 100,
        "update_mode": "live",
    },
}


def main() -> None:
    Tradeforce(config=CONFIG).run()


if __name__ == "__main__":
    main()
