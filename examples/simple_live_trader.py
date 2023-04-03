"""_summary_
"""

from tradeforce import Tradeforce

CONFIG = {
    "trader": {
        "id": 3,
        "run_live": True,
        "creds_path": "exchange_creds.cfg",
        "budget": 0,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
            "amount_invest_fiat": 1000,
            "investment_cap": 0,
            "buy_opportunity_factor": 0.10,
            "buy_opportunity_boundary": 0.05,
            "prefer_performance": 1,
            "hold_time_limit": 1000,
            "profit_factor": 1.10,
            "profit_ratio_limit": 1.01,
            "moving_window_hours": 160,
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
        "name": "bfx_history_docker_test",
        "exchange": "bitfinex",
        "base_currency": "USD",
        "candle_interval": "5min",
        "history_timeframe": "60days",
        "update_mode": "live",  # none, once or live
    },
}


# def main():
#     # To provide a subset, Tradeforce can also receive the optional argument "assets".
#     # This has to be a list of asset symbols.
#     # For example: Tradeforce(CONFIG, assets = ["BTC", "ETH", "XRP"]).run_new()
#     # Tradeforce(CONFIG).run_new()

#     tf = Tradeforce(CONFIG)
#     # asyncio.run(tf.run_new())

#     loop = asyncio.get_event_loop()
#     try:
#         loop.run_until_complete(tf.run_new())
#     except KeyboardInterrupt:
#         SystemExit("Bye!")
#         # Handle graceful shutdown on Ctrl+C


# if __name__ == "__main__":
#     main()


# async def main():
#     Tradeforce(CONFIG).exec()


# # asyncio.run(main())
# if __name__ == "__main__":
#     loop = asyncio.get_event_loop()

#     # Create a future and ensure that it's run within the event loop
#     future = loop.create_future()
#     asyncio.ensure_future(main(), loop=loop)

#     try:
#         loop.run_until_complete(future)
#     except KeyboardInterrupt:
#         print("KeyboardInterrupt received, stopping tasks...")

#     tasks = asyncio.all_tasks(loop)
#     for task in tasks:
#         task.cancel()

#     loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
#     loop.close()

tf = Tradeforce(CONFIG).run()
print(tf)
