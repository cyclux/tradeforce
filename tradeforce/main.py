""" /main.py

Tradeforce class orchestrates all modules
and provides a single interface to all features and various components.

"""

from __future__ import annotations


import importlib
import asyncio
import numpy as np
import optuna

from tradeforce.simulator import hyperparam_search
from tradeforce.config import Config

from tradeforce.logger import Logger
from tradeforce.backend import BackendMongoDB, BackendSQL
from tradeforce.market.history import MarketHistory
from tradeforce.market.updater import MarketUpdater
from tradeforce.exchange.api import ExchangeAPI
from tradeforce.exchange.websocket import ExchangeWebsocket
from tradeforce.trader import Trader
from tradeforce.utils import monkey_patch
from tradeforce import simulator

# Get current version from pyproject.toml
VERSION = importlib.metadata.version("tradeforce")

# TODO: add support for variable candle_interval
class Tradeforce:
    """Tradeforce class orchestrates all modules"""

    def __init__(self, config=None, assets=None):
        self.log = self._register_logger()
        self.assets_list_symbols = None if not assets or len(assets) == 0 else assets
        self.config = self._register_config(config)
        self.trader = self._register_trader()
        self.backend = self._register_backend()
        self.market_history = self._register_market_history()
        self.market_updater_api = self._register_updater()
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

    def _register_exchange_api(self) -> ExchangeAPI:
        return ExchangeAPI(root=self)

    def _register_exchange_ws(self) -> ExchangeWebsocket:
        return ExchangeWebsocket(root=self)

    def _register_trader(self) -> Trader:
        return Trader(root=self)

    #######################
    # Event loop handling #
    #######################

    def _exec_tasks(self, tasks: dict) -> "Tradeforce":
        """Run tasks in the event loop"""
        if tasks.get("load_history", True):
            asyncio.create_task(self.market_history.load_history())
        if self.config.update_mode == "live" and not self.config.is_sim:
            asyncio.create_task(self.exchange_ws.ws_run())
        if self.config.run_live and not self.config.is_sim:
            asyncio.create_task(self.exchange_ws.ws_priv_run())
        return self

    async def _async_exec_tasks(self, tasks: dict):
        """Convert _exec_tasks() to a coroutine"""
        self._exec_tasks(tasks)

    def _loop_handler(self, tasks: dict):
        """Helper to run the event loop and handle KeyboardInterrupts"""

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        # Run the tasks in the event loop, _exec_tasks() needs to be a coroutine
        asyncio.ensure_future(self._async_exec_tasks(tasks), loop=loop)
        try:
            loop.run_until_complete(future)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping tasks...")

        # When done / interrupted cancel all tasks
        loop_tasks = asyncio.all_tasks(loop)
        for task in loop_tasks:
            task.cancel()

        # Wait until all tasks are cancelled and return the gathered results
        loop.run_until_complete(asyncio.gather(*loop_tasks, return_exceptions=True))
        loop.close()
        return self

    #############
    # Run modes #
    #############

    def run(self) -> "Tradeforce":
        """Run the Tradeforce instance in normal mode.
        In Jupyter environment we can run the async tasks directly on the event loop.
        In normal CLI environment, we need additional measures to run the async tasks.
        Key goal is to keep the run() method synchronous and callable in both envs.
        """
        if is_jupyter():
            self._exec_tasks({})
        else:
            self._loop_handler({})
        return self

    async def _async_simulator_run(self):
        return simulator.run(self)

    def run_sim(self, pre_process=None, buy_strategy=None, sell_strategy=None) -> dict[str, int | np.ndarray]:
        """Run the Tradeforce instance in simulation mode.
        In Jupyter environment we need to async load_history() before running the simulator.
        """
        self.config.is_sim = True
        monkey_patch(self, pre_process, buy_strategy, sell_strategy)
        if not is_jupyter():
            asyncio.run(self.market_history.load_history())
        return simulator.run(self)

    def run_sim_optuna(
        self, optuna_config=None, pre_process=None, buy_strategy=None, sell_strategy=None
    ) -> optuna.Study:
        """Run the Tradeforce instance in simulation mode with hyperparameter optimization.
        In Jupyter environment we need to async load_history() before running the simulator.
        """
        self.config.is_sim = True
        monkey_patch(self, pre_process, buy_strategy, sell_strategy)
        if not is_jupyter():
            asyncio.run(self.market_history.load_history())
        return hyperparam_search.run(self, optuna_config)


def is_jupyter():
    """Check if we are running in a Jupyter environment"""
    try:
        get_ipython  # type: ignore
        return True
    except NameError:
        return False
