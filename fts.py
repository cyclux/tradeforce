"""_summary_
pip install numpy pandas pyarrow pymongo bitfinex-api-py websockets tensorflow-probability numexpr Bottleneck numba
"""
import asyncio
from fts_utils import connect_api
from fts_config import Config
from fts_backend import Backend
from fts_market_history import MarketHistory
from fts_market_updater import MarketUpdater
from fts_exchange_api import ExchangeAPI
from fts_exchange_websocket import ExchangeWebsocket
from fts_trader import Trader


class FastTradingSimulator:
    """_summary_"""

    def __init__(self, user_config, assets=None):
        self.assets_list_symbols = None if assets is None or len(assets) == 0 else assets
        self.config = self.register_config(user_config)
        self.backend = self.register_backend()
        self.market_history = self.register_market_history()
        self.market_updater_api = self.register_updater()
        self.api = self.connect_api()
        self.exchange_api = self.register_exchange_api()
        self.exchange_ws = self.register_exchange_ws()
        self.trader = self.register_trader()

    def register_config(self, user_config):
        config = Config(user_config)
        return config

    def register_backend(self):
        backend = Backend(fts_instance=self)
        return backend

    def register_updater(self):
        market_updater_api = MarketUpdater(fts_instance=self)
        return market_updater_api

    def register_market_history(self):
        market_history = MarketHistory(fts_instance=self)
        return market_history

    def connect_api(self):
        api = {}
        api["bfx_api_priv"] = connect_api(self.config.creds_path, "priv")
        api["bfx_api_pub"] = connect_api(self.config.creds_path, "pub")
        return api

    def register_exchange_api(self):
        exchange_api = ExchangeAPI(fts_instance=self)
        return exchange_api

    def register_exchange_ws(self):
        exchange_ws = ExchangeWebsocket(fts_instance=self)
        return exchange_ws

    def register_trader(self):
        if self.config.run_exchange_api:
            trader = Trader(fts_instance=self)
        else:
            trader = None
        return trader

    async def init(self):
        await self.market_history.load_history()
        if self.config.run_exchange_api:
            self.exchange_ws.ws_priv_run()
        return self

    def run(self):
        asyncio.run(self.init())
        if self.config.keep_updated:
            self.run_ws_updater()
        return self

    async def run_in_jupyter(self):
        # TODO: Transform to sync function. Maybe loop = asyncio.get_event_loop()
        # loop = asyncio.new_event_loop() or loop.run_until_complete(self._run_socket())
        await self.market_history.load_history()
        if self.config.run_exchange_api:
            self.exchange_ws.ws_priv_run()
        if self.config.keep_updated:
            self.run_ws_updater(run_in_jupyter=True)
        return self

    def run_ws_updater(self, run_in_jupyter=False):
        if not run_in_jupyter:
            loop_ws_updater = asyncio.new_event_loop()
            asyncio.set_event_loop(loop_ws_updater)
        self.exchange_ws.ws_run()
