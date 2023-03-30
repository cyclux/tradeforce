"""_summary_

"""

from __future__ import annotations
import os

import asyncio
import numpy as np
import optuna
from typing import TYPE_CHECKING
from tradeforce.simulator import hyperparam_search
from tradeforce.config import Config

from tradeforce.logger import Logger
from tradeforce.backend import BackendMongoDB, BackendSQL
from tradeforce.market.history import MarketHistory
from tradeforce.market.updater import MarketUpdater
from tradeforce.exchange.api import ExchangeAPI
from tradeforce.exchange.websocket import ExchangeWebsocket
from tradeforce.trader import Trader
from tradeforce.utils import connect_api, monkey_patch
from tradeforce import simulator

if TYPE_CHECKING:
    from bfxapi import Client  # type: ignore


# TODO: add support for variable candle_interval
class TradingEngine:
    """_summary_"""

    def __init__(self, config=None, assets=None):
        self.log = self._register_logger()
        self.assets_list_symbols = None if not assets or len(assets) == 0 else assets
        self.config = self._register_config(config)
        self.trader = self._register_trader()
        self.backend = self._register_backend()
        self.market_history = self._register_market_history()
        self.market_updater_api = self._register_updater()
        self.api = self._register_api()
        self.exchange_api = self._register_exchange_api()
        self.exchange_ws = self._register_exchange_ws()

    def __post__init__(self):
        self.log.info("Fast Trading Simulator Beta 0.4.0")

    #############################
    # Init and register modules #
    #############################

    def _register_logger(self):
        # self.logging = Logger()
        self.logging = Logger()
        log = self.logging.get_logger(__name__)
        return log

    def _register_config(self, user_config: dict) -> Config:
        return Config(root=self, config_input=user_config)

    def _register_backend(self) -> BackendMongoDB | BackendSQL | None:
        if self.config.dbms == "postgresql":
            return BackendSQL(root=self)
        if self.config.dbms == "mongodb":
            return BackendMongoDB(root=self)
        return None

    def _register_updater(self) -> MarketUpdater:
        return MarketUpdater(root=self)

    def _register_market_history(self) -> MarketHistory:
        working_dir = os.getcwd() if self.config.working_dir is None else self.config.working_dir
        return MarketHistory(root=self, path_current=working_dir)

    def _register_api(self) -> dict[str, Client | None]:
        api = {}
        if self.config.update_mode in ("once", "live"):
            api["bfx_api_pub"] = connect_api(self.config, "pub")
        if self.config.run_live:
            api["bfx_api_priv"] = connect_api(self.config, "priv")
        return api

    def _register_exchange_api(self) -> ExchangeAPI:
        return ExchangeAPI(root=self)

    def _register_exchange_ws(self) -> ExchangeWebsocket:
        return ExchangeWebsocket(root=self)

    def _register_trader(self) -> Trader:
        return Trader(root=self)

    async def _init(self, is_sim=False) -> "TradingEngine":
        await self.market_history.load_history()
        if self.config.run_live and not is_sim:
            self.exchange_ws.ws_priv_run()
        return self

    def _market_live_updates(self, run_in_jupyter=False) -> None:
        if not run_in_jupyter:
            loop_ws_updater = asyncio.new_event_loop()
            asyncio.set_event_loop(loop_ws_updater)
        self.exchange_ws.ws_run()

    #############
    # Run modes #
    #############

    def run(self) -> "TradingEngine":
        asyncio.run(self._init())
        if self.config.update_mode == "live":
            self._market_live_updates()
        return self

    def run_sim(self, pre_process=None, buy_strategy=None, sell_strategy=None) -> dict[str, int | np.ndarray]:
        monkey_patch(self, pre_process, buy_strategy, sell_strategy)
        asyncio.run(self._init(is_sim=True))
        return simulator.run(self)

    def run_sim_optuna(
        self, optuna_config=None, pre_process=None, buy_strategy=None, sell_strategy=None
    ) -> optuna.Study:
        monkey_patch(self, pre_process, buy_strategy, sell_strategy)
        asyncio.run(self._init(is_sim=True))
        return hyperparam_search.run(self, optuna_config)

    ##############################
    # Jupyter specific run modes #
    ##############################

    async def run_jupyter(self) -> "TradingEngine":
        await self.market_history.load_history()
        if self.config.run_live:
            self.exchange_ws.ws_priv_run()
        if self.config.update_mode == "live":
            self._market_live_updates(run_in_jupyter=True)
        return self

    async def run_sim_optuna_jupyter(
        self, optuna_config=None, pre_process=None, buy_strategy=None, sell_strategy=None
    ) -> optuna.Study:
        monkey_patch(self, pre_process, buy_strategy, sell_strategy)
        await self.market_history.load_history()
        return hyperparam_search.run(self, optuna_config)
