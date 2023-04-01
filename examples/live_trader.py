"""_summary_
"""

from tradeforce import Tradeforce

# config = {
#     "trader": {
#         "id": 3,
#         "run_live": False,
#         "creds_path": "exchange_creds.cfg",
#         "budget": 0,
#         "maker_fee": 0.10,
#         "taker_fee": 0.20,
#         "strategy": {
#             "amount_invest_fiat": 1000,
#             "investment_cap": 0,
#             "buy_opportunity_factor": 0.10,
#             "buy_opportunity_boundary": 0.05,
#             "prefer_performance": 1,
#             "hold_time_limit": 1000,
#             "profit_factor": 1.10,
#             "profit_ratio_limit": 1.01,
#             "moving_window_hours": 160,
#         },
#     },
#     "backend": {
#         "dbms": "postgresql",
#         "dbms_host": "docker_postgres",
#         "dbms_port": 5432,
#         "dbms_connect_db": "postgres",
#         "dbms_user": "postgres",
#         "dbms_pw": "postgres",
#         "local_cache": False,
#     },
#     "market_history": {
#         "name": "bfx_history_docker_test",
#         "exchange": "bitfinex",
#         "base_currency": "USD",
#         "candle_interval": "5min",
#         "history_timeframe": "60days",
#         "update_mode": "live",  # none, once or live
#     },
# }

# To provide a subset, Tradeforce can also receive the optional argument "assets".
# This has to be a list of asset symbols. For example: assets = ["BTC", "ETH", "XRP"]
Tradeforce().run()
