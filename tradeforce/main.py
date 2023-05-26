""" Module: ``tradeforce`` @ ``main.py``

TODO: Explain run modes.
"""

from __future__ import annotations

# import sys
from typing import TYPE_CHECKING, Callable
import importlib.metadata
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
    """Main application class.

    The Tradeforce class orchestrates all modules and components,
    providing methods to run the application in different modes.

    Args:
        config (:obj:`dict`, optional): A dictionary containing user-defined configuration settings.
                                        Either ``config`` or ``config_file`` has to be provided.

        config_file (:obj:`str`, optional): A path to a ``yaml`` configuration file. Can be used
                                            instead of passing a ``config`` dictionary.

        assets (:obj:`List[str]`, optional): A list of asset symbols to include. This has to be a
                                                subset of the assets available in the market history.
                                                If not provided, all assets will be included.
    """

    def __init__(self, config: dict | None = None, config_file: str | None = None, assets: list | None = None) -> None:
        """Initialize the Tradeforce instance

        with the provided configuration and asset symbols,
        and register the required modules.
        """
        self.log = self._register_logger()
        self.asset_symbols = [] if not assets else assets
        self.config = self._register_config(config, config_file)
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
            Logger instance.
        """
        self.logging = Logger()
        log = self.logging.get_logger(__name__)

        log.info(f"Tradeforce Beta {VERSION}")

        return log

    def _register_config(self, user_config: dict | None, config_file: str | None) -> Config:
        """Initialize and register the configuration.

        Args:
            user_config: An optional dictionary containing user-
                            defined configuration settings.

            config_file: An optional path to a ``yaml``
                            configuration file.

        Returns:
            Config: Provides access to config settings
                        to all modules it gets passed to.
        """
        config = Config(root=self, config_input=user_config, config_file=config_file)

        self.log.info("Current working directory: %s", config.working_dir)

        return config

    def _register_backend(self) -> BackendMongoDB | BackendSQL:
        """Initialize and register the database backend
            based on the configuration settings.

        Manages Backend database operations.

        Returns:
            Backend instance (BackendMongoDB | BackendSQL):
            Interface to the database backend.
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
            MarketUpdater instance.
        """
        return MarketUpdater(root=self)

    def _register_market_history(self) -> MarketHistory:
        """Initialize and register the market history.

        Manages the in-memory database as well as the
        Backend DB and provides methods to access and
        process historical market data.

        Returns:
            MarketHistory instance.
        """
        return MarketHistory(root=self)

    def _register_exchange_api(self) -> ExchangeAPI:
        """Initialize and register the exchange API.

        Manages authentication / connection to the exchange
        and provides methods to interact with the exchange API.

        Returns:
            ExchangeAPI instance.
        """
        return ExchangeAPI(root=self)

    def _register_exchange_ws(self) -> ExchangeWebsocket:
        """Initialize and register the exchange WebSocket.

        Manages the connection to the exchange WebSocket
        for real-time updates. Handles both public and
        private connections.

        Returns:
            ExchangeWebsocket instance.
        """
        return ExchangeWebsocket(root=self)

    def _register_trader(self) -> Trader:
        """Initialize and register the trader.

        Manages the live trading process and provides methods
        to buy and sell assets based on the provided strategy.

        Returns:
            Trader instance.
        """
        return Trader(root=self)

    # --------------------
    # Event loop handling
    # --------------------

    def _exec_tasks(self, tasks: dict) -> "Tradeforce":
        """Create and execute tasks async in the event loop.

        Which tasks to run is either determined by the input
        dictionary or in the user-defined configuration.

        Args:
            tasks: A dictionary containing tasks to be
                    executed, with task names as keys
                    and boolean values as flags.

        Returns:
            Current Tradeforce instance.
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

        The ensure_future() wrapper in _loop_handler()
        requires a coroutine.
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
        # future = loop.create_future()

        # Run the tasks in the event loop, _exec_tasks() needs to be a coroutine
        asyncio.ensure_future(self._async_exec_tasks(tasks), loop=loop)
        loop_tasks = asyncio.all_tasks(loop)
        try:
            loop.run_forever()

        # On exit / interrupt: cancel all tasks
        except KeyboardInterrupt:
            SystemExit("")
            # print("Exiting. Stopping tasks...")
            # loop.stop()
            # future.cancel()

            # loop_tasks = asyncio.all_tasks(loop)
            # print(loop_tasks)
            for task in loop_tasks:
                task.cancel()

            # Wait until all tasks are cancelled and return the gathered results
            # loop.run_until_complete(asyncio.gather(*loop_tasks, return_exceptions=True))

        finally:
            SystemExit("")
            # loop.stop()
            # loop.close()

        return self

    # ---------------------
    # Tradeforce run modes
    # ---------------------

    def run(self, load_history: bool = True) -> "Tradeforce":
        """Run Tradeforce in ``normal mode``.

        This run method enables primarily following use cases:

        - Dedicated market server:
            See `dedicated_market_server.py
            <https://github.com/cyclux/tradeforce/blob/master/examples/dedicated_market_server.py>`_
            in examples.

        - Live trading bot:
            See `live_trader_simple.py
            <https://github.com/cyclux/tradeforce/blob/master/examples/live_trader_simple.py>`_
            in examples.

        Warning:
            Use at your own risk! Tradeforce is currently in beta, and bugs may occur.
            Furthermore, there is no guarantee that strategies that have performed well
            in the past will continue to do so in the future.

        Note:
            Custom strategies are not yet available for the live trader.

        - Analysis in a Jupyter notebook:
            For analysis and visualization of simulation results.
            See `hyperparam_search_result_analysis.ipynb
            <https://github.com/cyclux/tradeforce/blob/master/examples/hyperparam_search_result_analysis.ipynb>`_
            in examples.

        Args:
            load_history: Whether to load the market history
                            on initialization. Default: True.
        """
        if _is_jupyter():
            self._exec_tasks({"load_history": load_history})
        else:
            self._loop_handler({"load_history": load_history})

        return self

    def _prepare_sim(
        self,
        pre_process: Callable | None = None,
        buy_strategy: Callable | None = None,
        sell_strategy: Callable | None = None,
    ) -> None:
        """Prepare the Tradeforce instance for simulation mode.
            Set the is_sim flag and optionally monkey patch.

        Some functions' logic adapts to the 'is_sim' flag, which
        is indicating we are running in "sim mode".

        If custom strategy or pre-process functions are provided,
        we need to "monkey patch" them -> Replacing the default ones.

        Args:
            pre_process: An optional custom pre-process function.
            buy_strategy: An optional custom buy strategy function.
            sell_strategy: An optional custom sell strategy function.
        """

        self.config.is_sim = True

        _monkey_patch(self, pre_process, buy_strategy, sell_strategy)

        if not _is_jupyter():
            asyncio.run(self.market_history.load_history())

    def run_sim(
        self,
        pre_process: Callable | None = None,
        buy_strategy: Callable | None = None,
        sell_strategy: Callable | None = None,
    ) -> dict[str, int | np.ndarray]:
        """Run Tradeforce in ``simulation mode``.

        Default strategies are used if no custom ones are provided
        (via parameters :py:obj:`pre_process`, :py:obj:`buy_strategy`, :py:obj:`sell_strategy`).
        For reference, how to define custom strategies, see:
        `simulator_custom.py
        <https://github.com/cyclux/tradeforce/blob/master/examples/simulator_custom.py>`_

        Note:
            In Jupyter environment before running a simulation, the
            history needs to be loaded via ``async load_history()``.
            For reference, see:
            `hyperparam_search_result_analysis.ipynb
            <https://github.com/cyclux/tradeforce/blob/master/examples/hyperparam_search_result_analysis.ipynb>`_

        Args:
            pre_process:   An optional function to pre-process market data.
            buy_strategy:  An optional function to determine buy signals.
            sell_strategy: An optional function to determine sell signals.

        Returns:
            :py:obj:`dict`: A dictionary containing the following keys:

            - **score** (:py:obj:`int`): ``mean(profit subsets) - std(profit subsets)``.

            - **trades** (:py:obj:`np.array`): Trading history, including buy and sell events.

            - **buy_log** (:py:obj:`np.array`): Buy log, containing the details of each buy event.
        """
        self._prepare_sim(pre_process, buy_strategy, sell_strategy)

        return simulator.run_train_val_split(self)

    def run_sim_optuna(
        self,
        optuna_config: dict,
        optuna_study: optuna.Study | None = None,
        pre_process: Callable | None = None,
        buy_strategy: Callable | None = None,
        sell_strategy: Callable | None = None,
    ) -> optuna.Study:
        """Run Tradeforce in simulation mode

        combined with Optuna hyperparameter optimization.

        Default strategies are used if no custom ones are provided.
        For reference, how to define custom strategies, see:
        `simulator_custom.py
        <https://github.com/cyclux/tradeforce/blob/master/examples/simulator_custom.py>`_

        Note:
            In Jupyter environment before running a simulation, the
            history needs to be loaded via ``async load_history()``.
            For reference, see:
            `hyperparam_search_result_analysis.ipynb
            <https://github.com/cyclux/tradeforce/blob/master/examples/hyperparam_search_result_analysis.ipynb>`_

        Args:
            optuna_config: A dictionary containing the Optuna configuration.
            optuna_study: An optional Optuna study object.
            pre_process: An optional function to pre-process market data.
            buy_strategy: An optional function to determine buy signals.
            sell_strategy: An optional function to determine sell signals.

        Returns:
            :py:obj:`optuna.Study`: An Optuna study object.
        """
        self._prepare_sim(pre_process, buy_strategy, sell_strategy)

        return hyperparam_search.run(self, optuna_config, optuna_study)


# -----------------
# Helper functions
# -----------------


def _is_jupyter() -> bool:
    """Check if we are running in a Jupyter environment.

    Returns:
        True if we are running in a Jupyter environment, False otherwise.
    """
    try:
        get_ipython  # type: ignore
        return True
    except NameError:
        return False


def _monkey_patch(
    root: Tradeforce,
    pre_process: Callable | None,
    buy_strategy: Callable | None,
    sell_strategy: Callable | None,
) -> None:
    """Monkey patch user-defined functions
        pre-process, buy and sell strategy, if provided.

    Args:
        root:          Tradeforce instance containing the Logger.

        pre_process:   A function to be used as pre_process:
                        Prepare data for simulation.

        buy_strategy:  A function to be used as buy_strategy:
                        Determine on which conditions to buy what.

        sell_strategy: A function to be used as sell_strategy:
                        Determine on which conditions to sell when.
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
