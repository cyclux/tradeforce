"""_summary_
"""


class Config:
    """_summary_
    SHIB, LEO, -> Not verified restriction
    """

    def __init__(self, config_input):
        self.update_history = config_input.get("update_history", False)
        self.run_exchange_api = config_input.get("run_exchange_api", False)
        self.keep_updated = config_input.get("keep_updated", False)

        self.exchange = config_input.get("exchange", "bitfinex")
        self.load_history_via = config_input.get("load_history_via", "feather").lower()
        self.dump_to_feather = config_input.get("dump_to_feather", False)
        self.asset_interval = config_input.get("asset_interval", "5min")
        self.base_currency = config_input.get("base_currency", "USD")
        self.history_timeframe = config_input.get("history_timeframe", "250h")
        self.backend = config_input.get("backend", "mongodb").lower()
        self.backend_host = config_input.get("backend_host", "localhost:1234")
        self.backend_user = config_input.get("backend_user", None)
        self.backend_password = config_input.get("backend_password", None)
        self.mongo_exchange_db = config_input.get("mongo_exchange_db", "bitfinex_db")
        self.mongo_collection = config_input.get("mongo_collection", "bfx_history")
        self.creds_path = config_input.get("creds_path", "")
        self.relevant_assets_cap = config_input.get("relevant_assets_cap", 100)

        self.trader_id = config_input.get("trader_id", 1)
        # self.gid = config_input.get("gid", 10**9)
        self.window = config_input.get("window", 20) * 60 // 5
        self.budget = float(config_input.get("budget", 0))
        self.buy_opportunity_factor = config_input.get("buy_opportunity_factor", 0.0)
        self.buy_opportunity_boundary = config_input.get("buy_opportunity_boundary", 0.02)
        self.buy_opportunity_factor_min = self.buy_opportunity_factor - self.buy_opportunity_boundary
        self.buy_opportunity_factor_max = self.buy_opportunity_factor + self.buy_opportunity_boundary
        self.profit_factor = config_input.get("profit_factor", 0.05)
        self.profit_ratio_limit = config_input.get("profit_ratio_limit", 0.9)
        self.prefer_performance = config_input.get("prefer_performance", "center")
        self.amount_invest_fiat = config_input.get("amount_invest_fiat", None)
        self.amount_invest_relative = config_input.get("amount_invest_relative", None)
        self.max_buy_per_asset = config_input.get("max_buy_per_asset", 1)
        self.hold_time_limit = config_input.get("hold_time_limit", 20000)
        self.buy_limit_strategy = config_input.get("buy_limit_strategy", False)
        self.exchange_fee = config_input.get("exchange_fee", 0.15)
        self.use_backend = config_input.get("use_backend", True)
        self.is_simulation = config_input.get("is_simulation", False)

        self.assets_excluded = [
            "UDC",
            "UST",
            "EUT",
            "EUS",
            "MIM",
            "PAX",
            "DAI",
            "TSD",
            "TERRAUST",
            "LEO",
            "WBT",
            "RBT",
            "ETHW",
            "SHIB",
            "ETH2X",
        ]
