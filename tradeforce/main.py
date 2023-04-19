"""
Tradeforce is a highly performant trading framework for backtesting and hyperparameter optimization.
However, it can also be used for live trading.

Key features:
- Highly performant simulations enabled by Numba
- Customizable trading strategies
- Dedicated market server: 100+ simultanious Websocket connections
- Backend support for PostgresSQL and MongoDB
- Hyperparameter optimization
- Live trading capabilities
- Easy and flexible deployment with Docker
- Scalable to cloud environments with Kubernetes
- Juptyer Notebook support for result visualization and analysis



It lets you test trading strategies on historical data and optimize them with Optuna.

"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import importlib
import asyncio
import numpy as np
import optuna

from tradeforce.simulator import hyperparam_search
from tradeforce.config import Config

import tradeforce.simulator.default_strategies as strategies
from tradeforce.logger import Logger
from tradeforce.backend import BackendMongoDB, BackendSQL
from tradeforce.market.history import MarketHistory
from tradeforce.market.updater import MarketUpdater
from tradeforce.exchange.api import ExchangeAPI
from tradeforce.exchange.websocket import ExchangeWebsocket
from tradeforce.trader import Trader

# Prevent circular import for type checking
if TYPE_CHECKING:
    import logging

# from tradeforce.utils import monkey_patch
from tradeforce import simulator

# Get current version from pyproject.toml
VERSION = importlib.metadata.version("tradeforce")


class Tradeforce:
    """The Tradeforce class orchestrates all modules and components

    of the trading framework, providing methods to run the application
    in different modes and manage its components.

    Attributes:
        log (logging.Logger):                  The logger instance used for logging.
        asset_symbols (List[str]):             A list of asset symbols to be used in the platform.
        config (Config):                       The configuration object for the trading platform.
        trader (Trader):                       The trader object responsible for managing trades.
        backend (BackendMongoDB | BackendSQL): The database backend object for storing data.
        market_history (MarketHistory):        The market history object for handling historical data.
        market_updater_api (MarketUpdater):    The market updater object for updating market data.
        exchange_api (ExchangeAPI):            The exchange API object for interacting with the exchange.
        exchange_ws (ExchangeWebsocket):       The exchange WebSocket object for real-time updates.

    Params:
        config: A dictionary containing user-defined configuration settings.
        assets: A list of asset symbols to be used. If not provided, all assets will be used.
    """

    def __init__(self, config: dict | None = None, assets: list | None = None) -> None:
        """Initialize the Tradeforce instance

        with the provided configuration and asset symbols,
        and register the required modules.

        Params:
            config: A dictionary containing user-defined configuration settings.
            assets: A list of asset symbols to be used in the platform.
                        If not provided, all assets will be used.
        """
        self.log = self._register_logger()
        self.asset_symbols = [] if not assets else assets
        self.config = self._register_config(config)
        self.trader = self._register_trader()
        self.backend = self._register_backend()
        self.market_history = self._register_market_history()
        self.market_updater_api = self._register_updater()
        self.exchange_api = self._register_exchange_api()
        self.exchange_ws = self._register_exchange_ws()

    # --------------------------
    # Init and register modules
    # --------------------------

    def _register_logger(self) -> logging.Logger:
        """Initialize and register the logger.

        Returns:
            The logger instance.
        """
        self.logging = Logger()
        log = self.logging.get_logger(__name__)

        log.info(f"Tradeforce Beta {VERSION}")

        return log

    def _register_config(self, user_config: dict | None) -> Config:
        """Initialize and register the configuration.

        Params:
            user_config: An optional dictionary containing user-
                            defined configuration settings.

        Returns:
            Config object: Provides access to config settings
                            to all modules.
        """
        config = Config(root=self, config_input=user_config)

        self.log.info("Current working directory: %s", config.working_dir)

        return config

    def _register_backend(self) -> BackendMongoDB | BackendSQL:
        """Initialize and register the database backend
        -> based on the configuration settings.

        Manages Backend database operations.

        Returns:
            The (BackendMongoDB | BackendSQL) object
        """
        if self.config.dbms == "postgresql":
            return BackendSQL(root=self)

        elif self.config.dbms == "mongodb":
            return BackendMongoDB(root=self)

        else:
            raise ValueError(f"DBMS {self.config.dbms} not supported.")

    def _register_updater(self) -> MarketUpdater:
        """Initialize and register the market updater.

        Manages the market data update process
        between MarketHistory, ExchangeAPI and Backend

        Returns:
            The MarketUpdater object.
        """
        return MarketUpdater(root=self)

    def _register_market_history(self) -> MarketHistory:
        """Initialize and register the market history.

        Manages the in-memory database as well as the
        Backend DB and provides methods to access and
        process historical market data.

        Returns:
            The MarketHistory object.
        """
        return MarketHistory(root=self)

    def _register_exchange_api(self) -> ExchangeAPI:
        """Initialize and register the exchange API.

        Manages authentication / connection to the exchange
        and provides methods to interact with the exchange API.

        Returns:
            The ExchangeAPI object.
        """
        return ExchangeAPI(root=self)

    def _register_exchange_ws(self) -> ExchangeWebsocket:
        """Initialize and register the exchange WebSocket.

        Manages the connection to the exchange WebSocket
        for real-time updates. Handles both public and
        private connections.

        Returns:
            The ExchangeWebsocket object.
        """
        return ExchangeWebsocket(root=self)

    def _register_trader(self) -> Trader:
        """Initialize and register the trader.

        Manages the live trading process and provides methods
        to buy and sell assets based on the provided strategy.

        Returns:
            The Trader object.
        """
        return Trader(root=self)

    # --------------------
    # Event loop handling
    # --------------------

    def _exec_tasks(self, tasks: dict) -> "Tradeforce":
        """Create and execute tasks async in the event loop.

        Which tasks to run is either determined by the input
        dictionary or in the user-defined configuration.

        Params:
            tasks: A dictionary containing tasks to be executed,
            with task names as keys and boolean values as flags.

        Returns:
            The current Tradeforce instance.
        """

        if tasks.get("load_history", True):
            asyncio.create_task(self.market_history.load_history())

        if self.config.update_mode == "live" and not self.config.is_sim:
            asyncio.create_task(self.exchange_ws.ws_run())

        if self.config.run_live and not self.config.is_sim:
            asyncio.create_task(self.exchange_ws.ws_priv_run())

        return self

    async def _async_exec_tasks(self, tasks: dict) -> None:
        """Convert _exec_tasks() to a coroutine
        -> The ensure_future() wrapper in _loop_handler() requires a coroutine.
        """
        self._exec_tasks(tasks)

    def _loop_handler(self, tasks: dict) -> "Tradeforce":
        """Handle event loop for running tasks in CLI environment.

        This methods handles the CLI / 'non-Jupyter notebook' use case:
            Set up and run the event loop for the specified tasks, handle
            KeyboardInterrupts, and properly clean up the tasks upon
            completion or interruption.

        Note:
            In Jupyter environment we can run the async tasks directly
            on the event loop. However, in 'normal' CLI environment,
            we need additional measures to run the async tasks. Key goal
            is to keep the run() method synchronous and callable in
            both environments.
        """

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        # Run the tasks in the event loop, _exec_tasks() needs to be a coroutine
        asyncio.ensure_future(self._async_exec_tasks(tasks), loop=loop)

        try:
            loop.run_until_complete(future)

        # On exit / interrupt: cancel all tasks
        except KeyboardInterrupt:
            print("Exiting. Stopping tasks...")

            future.cancel()

            loop_tasks = asyncio.all_tasks(loop)
            for task in loop_tasks:
                task.cancel()

            # Wait until all tasks are cancelled and return the gathered results
            loop.run_until_complete(asyncio.gather(*loop_tasks, return_exceptions=True))

        finally:
            loop.close()

        return self

    # ---------------------
    # Tradeforce run modes
    # ---------------------

    def run(self) -> "Tradeforce":
        """Run the Tradeforce instance in normal mode."""
        if _is_jupyter():
            self._exec_tasks({"load_history": True})
        else:
            self._loop_handler({"load_history": True})
        return self

    async def _async_simulator_run(self) -> dict[str, int | np.ndarray]:
        return simulator.run(self)

    def run_sim(
        self,
        pre_process: Callable | None = None,
        buy_strategy: Callable | None = None,
        sell_strategy: Callable | None = None,
    ) -> dict[str, int | np.ndarray]:
        """Run the Tradeforce instance in simulation mode.

        In Jupyter environment we need to async load_history()
        before running the simulator.
        """
        self.config.is_sim = True
        _monkey_patch(self, pre_process, buy_strategy, sell_strategy)
        if not _is_jupyter():
            asyncio.run(self.market_history.load_history())
        return simulator.run(self)

    def run_sim_optuna(
        self,
        optuna_config: dict,
        pre_process: Callable | None = None,
        buy_strategy: Callable | None = None,
        sell_strategy: Callable | None = None,
    ) -> optuna.Study:
        """Run the Tradeforce instance in simulation mode
        -> combined with Optuna hyperparameter optimization.

        Note:
            In the Jupyter environment we need to async load_history()
            before running the simulator.
        """
        self.config.is_sim = True
        _monkey_patch(self, pre_process, buy_strategy, sell_strategy)
        if not _is_jupyter():
            asyncio.run(self.market_history.load_history())
        return hyperparam_search.run(self, optuna_config)


# -----------------
# Helper functions
# -----------------


def _is_jupyter() -> bool:
    """Check if we are running in a Jupyter environment."""
    try:
        get_ipython  # type: ignore
        return True
    except NameError:
        return False


def _monkey_patch(
    root: Tradeforce, pre_process: Callable | None, buy_strategy: Callable | None, sell_strategy: Callable | None
) -> None:
    """Monkey patch user defined pre_process function or buy and sell strategies if provided.

    Params:
        root:          The Tradeforce main instance.
        pre_process:   A function to be used as pre_process: Prepare data for simulation.
        buy_strategy:  A function to be used as buy_strategy: Determine on which conditions to buy what.
        sell_strategy: A function to be used as sell_strategy: Determine on which conditions to sell when.
    """

    if pre_process:
        root.log.info("Custom pre_process loaded")
        strategies.pre_process = pre_process
    else:
        root.log.info("Default pre_process loaded")

    if buy_strategy:
        root.log.info("Custom buy_strategy loaded")
        strategies.buy_strategy = buy_strategy
    else:
        root.log.info("Default buy_strategy loaded")

    if sell_strategy:
        root.log.info("Custom sell_strategy loaded")
        strategies.sell_strategy = sell_strategy
    else:
        root.log.info("Default buy_strategy loaded")
