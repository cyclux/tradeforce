""" /main.py

Tradeforce class orchestrates all modules
and provides a single interface to all features and various components.

"""

from __future__ import annotations


import importlib
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

# Get current version from pyproject.toml
VERSION = importlib.metadata.version("tradeforce")

# TODO: add support for variable candle_interval
class Tradeforce:
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

    #############################
    # Init and register modules #
    #############################

    def _register_logger(self):
        self.logging = Logger()
        log = self.logging.get_logger(__name__)
        log.info(f"Fast Trading Simulator Beta {VERSION}")
        return log

    def _register_config(self, user_config: dict) -> Config:
        config = Config(root=self, config_input=user_config)
        self.log.info("Current working directory: %s", config.working_dir)
        return config

    def _register_backend(self) -> BackendMongoDB | BackendSQL | None:
        if self.config.dbms == "postgresql":
            return BackendSQL(root=self)
        if self.config.dbms == "mongodb":
            return BackendMongoDB(root=self)
        return None

    def _register_updater(self) -> MarketUpdater:
        return MarketUpdater(root=self)

    def _register_market_history(self) -> MarketHistory:
        return MarketHistory(root=self)

    def _register_api(self) -> dict[str, Client | None]:
        api = {}
        if self.config.update_mode in ("once", "live"):
            api["bfx_api_pub"] = connect_api(self, "pub")
        if self.config.run_live:
            api["bfx_api_priv"] = connect_api(self, "priv")
        return api

    def _register_exchange_api(self) -> ExchangeAPI:
        return ExchangeAPI(root=self)

    def _register_exchange_ws(self) -> ExchangeWebsocket:
        return ExchangeWebsocket(root=self)

    def _register_trader(self) -> Trader:
        return Trader(root=self)

    #############
    # Run modes #
    #############

    def exec(self) -> "Tradeforce":
        asyncio.create_task(self.market_history.load_history())
        if self.config.update_mode == "live":
            asyncio.create_task(self.exchange_ws.ws_run())
        if self.config.run_live:
            asyncio.create_task(self.exchange_ws.ws_priv_run())
        return self

    async def async_exec(self):
        self.exec()

    def loop_handler(self):
        loop = asyncio.get_event_loop()
        # Create a future and ensure that it's run within the event loop
        future = loop.create_future()
        asyncio.ensure_future(self.async_exec(), loop=loop)

        try:
            loop.run_until_complete(future)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping tasks...")

        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()

        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.close()
        return self

    def run(self) -> "Tradeforce":
        try:
            get_ipython  # type: ignore
            is_jupyter = True
        except NameError:
            is_jupyter = False

        if is_jupyter:
            self.exec()
        else:
            self.loop_handler()
        return self

    def run_sim(self, pre_process=None, buy_strategy=None, sell_strategy=None) -> dict[str, int | np.ndarray]:
        monkey_patch(self, pre_process, buy_strategy, sell_strategy)
        asyncio.run(self.market_history.load_history())
        return simulator.run(self)

    def run_sim_optuna(
        self, optuna_config=None, pre_process=None, buy_strategy=None, sell_strategy=None
    ) -> optuna.Study:
        monkey_patch(self, pre_process, buy_strategy, sell_strategy)
        asyncio.run(self.market_history.load_history())
        return hyperparam_search.run(self, optuna_config)

    ##############################
    # Jupyter specific run modes #
    ##############################

    async def run_sim_optuna_jupyter(
        self, optuna_config=None, pre_process=None, buy_strategy=None, sell_strategy=None
    ) -> optuna.Study:
        monkey_patch(self, pre_process, buy_strategy, sell_strategy)
        await self.market_history.load_history()
        return hyperparam_search.run(self, optuna_config)
