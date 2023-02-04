"""_summary_
"""

from fts import FastTradingSimulator

config = {
    # trading_strategy
    "trader_id": 2,
    "use_backend": True,
    "is_simulation": False,
    "amount_invest_fiat": 100,
    # "amount_invest_relative": None,
    "buy_limit_strategy": False,
    "buy_opportunity_factor": 0.16,
    "buy_opportunity_boundary": 0.01,
    "prefer_performance": "positive",
    "max_buy_per_asset": 1,
    "hold_time_limit": 12000,
    "profit_factor": 1.70,
    "profit_ratio_limit": 1,
    "window": 160,
    "exchange_fee": 0.20,
    "budget": 0,
    # history_config
    "asset_interval": "5min",
    "history_timeframe": "100days",
    "base_currency": "USD",
    "exchange": "bitfinex",
    "load_history_via": "API",
    "backend": "mongodb",
    "backend_host": "localhost:1234",
    "mongo_collection": "bfx_history_test",
    "creds_path": "exchange_creds.cfg",
    "update_history": False,
    "run_exchange_api": True,
    "keep_updated": True,
}

# assets = ["BTC", "ETH", "XMR"]

fts = FastTradingSimulator(config).run()
