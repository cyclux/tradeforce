""" config.py

Module: tradeforce.config
-------------------------

Config module containing the Config class for Tradeforce.

Loads config dict or config.yaml file and stores it in a Config class
which can get passed to all other modules, classes and functions.

The Config class handles various settings and configurations for Tradeforce,
including data handling settings, exchange settings, trader and strategy
settings, and simulator configurations. It is responsible for loading, updating,
and storing these settings, as well as providing a method to represent
the configuration in a dictionary format.

Attributes:
    working_dir (str): Directory for storing configuration and data files.

    candle_interval (str): Time interval between candlesticks.

    exchange (str): Exchange to be used for trading and data fetching.

    fetch_init_timeframe_days (int): Initial timeframe for
        fetching historical data.

    dbms_history_entity_name (str): Entity name for storing historical data.

    local_cache (bool): Whether to cache historical data
        locally for faster access.

    credentials_path (str): Path to exchange API credentials file.

    run_live (bool): Whether to run in live trading mode or not.

    trader_id (int): Unique identifier for a trader instance.

    budget (float): Initial trading budget in base currency.

    amount_invest_per_asset (float): Amount to invest per asset
        upon purchase.

    relevant_assets_cap (int): Maximum number of assets to consider
        for trading or simulation.

    min_amount_candles (int): Minimum number of candles required
                                for an asset to be considered relevant.

    max_candle_sparsity (int): Maximum "sparcity" constraint of candles
                                to be considered as a relevant asset.

    moving_window_hours (int): Timeframe for moving window used
        in the sim to calulate the price performance score.

    moving_window_increments (int): Number of candle increments
        in the moving window. Derived from candle_interval
        and moving_window_hours.

    buy_signal_score (float): Performance score of an asset
        within the moving window.

    buy_signal_boundary (float): Range of acceptable
        buy_signal_score for purchasing assets.

    buy_signal_preference (int): Preference for buying assets
        with positive, negative, or closest performance.

    profit_factor_target (float): Target profit factor to sell assets.

    profit_factor_target_min (float): Minimum profit factor
        after _hold_time_increments reached.

    hold_time_days (int): Number of days to hold an asset before
        reducing to profit_factor_target_min.

    _hold_time_increments (int): Conversion of hold_time_days to
        increments based on the candle_interval.

    max_buy_per_asset (int): Maximum number of times a single
        asset can be bought.

    investment_cap (float): Maximum amount of budget that can be
        invested in the market simultaneously.

    maker_fee (float): Maker fee percentage for the exchange.

    taker_fee (float): Taker fee percentage for the exchange.

    subset_size_days (int): Size of subsets within the
        entire market candle history.

    _subset_size_increments (int): Conversion of subset_size_days
        to increments based on the candle_interval.

    subset_amount (int): Number of subsets to create from
        the entire market history.

    train_val_split_ratio (float): Ratio of training to validation
                                    dataset size.

    assets_excluded (list of symbols): List of assets to
        be excluded from trading.

    use_dbms (bool): Whether to use a DBMS for data storage.

    is_sim (bool): Whether the current run is a simulation or not.

"""


from __future__ import annotations
import os
from pathlib import Path
import yaml
from typing import TYPE_CHECKING

from tradeforce.utils import candle_interval_to_min

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import Tradeforce

# Paths to check for config.yaml/yml file
config_paths = ["config.yaml", "config.yml", "config/config.yaml", "config/config.yml"]


def _load_yaml_config(config_path: str) -> dict:
    """Load the YAML configuration file.

    Attempt to load a YAML configuration file from the given file path.

    Params:
        config_path: A string representing the path to the configuration file.

    Returns:
        Dict containing the loaded configuration data if the file is
        found and successfully loaded, otherwise an empty dictionary.
    """
    try:
        with open(config_path, "r", encoding="utf8") as stream:
            yaml_config = yaml.safe_load(stream)
    except FileNotFoundError:
        return {}
    return dict(yaml_config)


def _try_config_paths(config_paths: list) -> tuple[dict, str]:
    """Attempt to load configuration data from a list of file paths.

    Iterate over the provided list of configuration file paths and try to
    load the YAML configuration file from each path. Stop once a valid
    configuration file is found and loaded.

    Params:
        config_paths: A list of strings, each representing a file path
        to a configuration file.

    Returns:
        A tuple containing the loaded configuration data as a dictionary
        and the absolute file path of the configuration file. If no valid
        configuration file is found, return an empty dictionary and an
        empty string.
    """
    while len(config_paths) > 0:
        current_config_path = config_paths.pop(0)
        congig_path_absolute = os.path.join(os.path.abspath(""), current_config_path)
        config_input = _load_yaml_config(congig_path_absolute)
        if config_input:
            break
    return config_input, congig_path_absolute


def _flatten_dict(input_config_dict: dict) -> dict:
    """Flatten a nested dictionary to a single-level dictionary.

    Recursively flatten the input dictionary, so that nested keys are combined
    into a single key in the output dictionary.

    This is useful to disregard the structure of the nested configuration.

    Note:
        Configuration keys need to be unique!

    Params:
        input_config_dict: A dictionary potentially containing nested dictionaries.

    Returns:
        A new dictionary with the same keys and values as the input dictionary,
        but with all nested dictionaries flattened to a single level.
    """
    output_dict: dict = {}

    def flatten(input_dict: dict, key: str = "") -> None:
        if isinstance(input_dict, dict):
            for key in input_dict:
                flatten(input_dict[key], key)
        else:
            output_dict[key] = input_dict

    flatten(input_config_dict)
    return dict(output_dict)


def _hours_to_increments(hours: int, candle_interval: str) -> int:
    """Convert hours to increments corresponding to candle_interval.

    One increment is one candle timestep. e.g. 1h = 60min = 12 candles
    """
    candle_interval_min = candle_interval_to_min(candle_interval)
    increments = int(hours * 60 // candle_interval_min)
    if increments <= 0:
        increments = -1
    return increments


class Config:
    """The Config class is used to load and store the user config.

    It gets passed to all other classes and allows us to access them via self.config.<config_key>
    Provides default values to fall back on if param is not specified in user config.
    """

    def __init__(self, root: Tradeforce, config_input: dict | None, config_file: str | None):
        self.log = root.logging.get_logger(__name__)
        self.config_input: dict = self.load_config(config_input, config_file)

        # Set all config keys as attributes of the Config class
        # This allows us to access them via self.config.<config_key> in any module
        self.set_config_as_attr(self.config_input)

        # ------------
        # Core setup
        # ------------

        self.working_dir = Path(self.config_input.get("working_dir", os.path.abspath("")))
        """working_dir: Path to working directory
        The path to the working directory. Default is the directory where Tradeforce is executed.
        """

        self.check_db_consistency = self.config_input.get("check_db_consistency", True)
        """check_db_consistency: bool

        Determines whether to check the in-memory database consistency by verifying no missing candles between
        the start and end timestamps. Useful to disable in hyperparameter search scenarios for performance.
        """

        self.check_db_sync = self.config_input.get("check_db_sync", True)
        """check_db_sync: bool

        Checks whether the in-memory database is in sync with the backend database.
        If not, it will try to sync bi-directionally by exchanging missing candles
        between the backend and in-memory database.
        During simulation mode, this is usually not necessary and can be disabled for performance.
        """

        # ---------------
        # Logging Levels
        # ---------------

        self.log_level_ws_live = self.config_input.get("log_level_ws_live", "ERROR").upper()
        """log_level_ws_live: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

        Log level for the private websocket connection during live trading.

        """

        self.log_level_ws_update = self.config_input.get("log_level_ws_update", "ERROR").upper()
        """log_level_ws_update: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

        Log level for the public websocket connection (candle/price updater).
        """

        # -------------------------
        # Market history & Updates
        # -------------------------

        self.exchange = self.config_input.get("exchange", "bitfinex")
        """exchange: "bitfinex"

        The name of the exchange to use. Currently, only Bitfinex is supported.
        """

        self.base_currency = self.config_input.get("base_currency", "USD")
        """base_currency: "USD"

        The base currency to trade against, such as "USD" or "BTC".
        """

        self.candle_interval = self.config_input.get("candle_interval", "5min")
        """candle_interval: str: "1min", "5min", "15min", "30min", "1h", "3h", "6h", "12h", "1D", "1W"

        The interval for market history and update candles.
        Shorter intervals result in more candles and less precision,
        while longer intervals are more efficient but less precise.
        """

        self.fetch_init_timeframe_days = self.config_input.get("fetch_init_timeframe_days", 120)
        """fetch_init_timeframe_days: currently needs to be in "days".

        The number of days in the past to load the candle history from.
        """

        self.force_source = self.config_input.get("force_source", "none").lower()
        """force_source: "none", "local_cache", "backend", "api"

        Specifies a particular data source for the candle history:
        "none" Search in order: local_cache, backend, api.
        "local_cache": "arrow" dump of in-memory DB (working_dir/data).
        "backend": Fetch history from DB.
        "api": Fetch history from exchange via Websocket.
        """

        self.update_mode = self.config_input.get("update_mode", "none").lower()
        """update_mode: "none", "once", "live"

        Update candle history of the DB to the current time.
        none: Do not update
        once: Only update statically once via REST API
        live: Update continuously via Websocket

        Determines how to update the candle history of the database:
        "none": Do not update.
        "once": Static update via REST API.
        "live": Continuous update via Websocket.
        """

        # ---------------------
        # Database Connection
        # ---------------------

        self.dbms = self.config_input.get("dbms", "postgres").lower()
        """dbms : "postgres", "mongodb"

        The "DataBase Management System" to use.
        Currently, Postgres and MongoDB are supported.
        Choose Postgres for full functionality, including hyperparameter search via optuna.
        """

        self.dbms_host = self.config_input.get("dbms_host", "docker_postgres")
        """dbms_host: str

        The fully qualified domain name (FQDN) of the database management system,
        such as "localhost" or a remote address. "docker_postgres" is the default for the docker container.
        """

        self.dbms_port = int(self.config_input.get("dbms_port", 5432))
        """dbms_port: int

        The port number of the DBMS.
        """

        self.dbms_user = self.config_input.get("dbms_user", "postgres")
        """ dbms_user: str

        Username of the dbms_db. "postgres" is the default user of the postgres docker container.
        """

        self.dbms_pw = self.config_input.get("dbms_pw", "postgres")
        """dbms_pw: str

        The password for the database. "postgres" is the default password of the postgres docker container.
        """

        self.dbms_connect_db = self.config_input.get("dbms_connect_db", "postgres")
        """dbms_connect_db: "postgres"

        The name of an existing database to connect to when creating a new database:
        Specific to SQL DBMS like Postgres.
        """

        self.dbms_db = self.config_input.get("dbms_db", f"{self.exchange}_db")
        """dbms_db: str

        The name of the database to use. Gets created if it does not exist.
        """

        self.dbms_history_entity_name = self.config_input.get(
            "name", f"{self.exchange}_history_{self.fetch_init_timeframe_days}days"
        )
        """dbms_history_entity_name: str

        The name of the entity (TABLE in Postgres or COLLECTION in MongoDB) to store candle history data.
        """

        self.local_cache = self.config_input.get("local_cache", True)
        """local_cache: bool

        Whether to save the in-memory database to a local arrow file for faster loading in subsequent runs.
        Gets saved in working_dir/data.
        """

        # -------------------
        # Exchange Settings
        # -------------------

        self.credentials_path = self.config_input.get("credentials_path", self.working_dir)
        """credentials_path: str

        The path to the exchange credentials file for authenticating the private websocket connection (Live trader).
        """

        self.run_live = self.config_input.get("run_live", False)
        """run_live: bool

        Whether to run Tradeforce in live trading mode, executing the configured strategy on the exchange.
        Needs valid credentials in credentials_path.
        CAUTION: This is NOT a backtest / simulation. The strategy is executed on the exchange.
        """

        # ----------------------------
        # Trader & Strategy Settings
        # ----------------------------

        self.trader_id = int(self.config_input.get("id", 1))
        """trader_id: int

        Unique identifier for a trader instance, useful when running multiple trading strategies on the same database.
        """

        self.budget = float(self.config_input.get("budget", 0))
        """budget: float

        Initial trading budget in base currency (e.g., USD) for the trader, used primarily for simulations.
        In live trading, the budget is derived from the base currency balance in the exchange wallet.
        """

        self.amount_invest_per_asset = self.config_input.get("amount_invest_per_asset", 0.0)
        """amount_invest_per_asset: float

        Amount of base currency (usually fiat, e.g., USD) to invest in each asset upon purchase.
        """

        self.relevant_assets_cap = int(self.config_input.get("relevant_assets_cap", 100))
        """relevant_assets_cap: int

        Maximum number of assets to consider for trading or simulations, applied after filtering for relevant assets.
        """
        # Constraints for relevant assets

        self.min_amount_candles = int(self.config_input.get("min_amount_candles", 2000))
        """min_amount_candles: int

        Minimum number of candles required for an asset to be considered relevant.
        "Relevant assets" get filtered while initially fetching the candle history via API.
        """

        self.max_candle_sparsity = int(self.config_input.get("max_candle_sparsity", 500))
        """max_candle_sparsity: int

        Maximum "sparcity" constraint of candles to be considered as a relevant asset.
        "Relevant assets" get filtered while initially fetching the candle history via API.
        """

        self.moving_window_hours = self.config_input.get("moving_window_hours", 20)
        """moving_window_hours: int

        Timeframe (in hours) of the moving window used in the buy_strategy,
        to calculate the buy_signal_score.
        """

        self.moving_window_increments = _hours_to_increments(self.moving_window_hours, self.candle_interval)
        """moving_window_increments: int

        Number of candle increments in the moving window, derived from candle_interval and moving_window_hours.
        Not intended for manual configuration.
        """

        # Default trading strategy parameters

        self.buy_signal_score = self.config_input.get("buy_signal_score", 0.0)
        """buy_signal_score: float

        Performance score of an asset within the moving window.
        In the default strategy implementation typically calculated
        as the "sum of percentage changes" within the window.
        """

        self.buy_signal_boundary = self.config_input.get("buy_signal_boundary", 0.02)
        """buy_signal_boundary: float

        Lower and upper limit for the buy_signal_score,
        defining a range of acceptable scores for purchasing assets.
        e.g. boundary 0.02 means +/-0.02 the buy_signal_score to be considered for purchase.
        """

        self.buy_signal_preference = self.config_input.get("buy_signal_preference", 1)
        """buy_signal_preference: -1, 0, 1

        Preference for buying assets with performance
        positive (1) -> decending, higher values prefered,
        negative (-1) -> ascending, lower values prefered,
        or closest to the buy_signal_score (0).
        """

        self.profit_factor_target = self.config_input.get("profit_factor_target", 0.05)
        """profit_factor_target: float, positive

        Target profit factor for each asset.
        e.g. 1.05 represents a target of 5% profit on each trade.
        """

        self.profit_factor_target_min = self.config_input.get("profit_factor_target_min", 1.01)
        """profit_factor_target_min: float

        Minimum profit factor after _hold_time_increments,
        allowing the trader to sell assets at lower profits or even losses.
        """

        self.hold_time_days = self.config_input.get("hold_time_days", 4)
        """hold_time_days: int

        Number of days to hold an asset before reducing
        the profit_factor to profit_factor_min.
        """

        self._hold_time_increments = _hours_to_increments(24 * self.hold_time_days, self.candle_interval)
        """_hold_time_increments: int

        Conversion of hold_time_days to increments based on the candle_interval,
        used as an internal time reference.
        """

        # TODO: implement max_buy_per_asset > 1
        self.max_buy_per_asset = self.config_input.get("max_buy_per_asset", 1)
        """max_buy_per_asset: int -> currently only 1 supported

        Maximum number of times a single asset can be bought,
        useful for managing repeated purchases of an asset with a dropping price.
        """

        # TODO: implement investment_cap for live trader

        self.investment_cap = self.config_input.get("investment_cap", 0.0)
        """investment_cap: float, currently only implemented for simulations

        0.0 -> unlimited investment, no cap. All potential profits are reinvested.

        Maximum amount of budget (in base currency) that can be invested in the market simultaneously,
        useful for limited reinvestment of profits.
        """

        self.maker_fee = self.config_input.get("maker_fee", 0.10)
        """maker_fee: float

        Maker fee percentage for the exchange, used to calculate fees in simulated trades.
        In live trading, actual fees from the exchange API are used.
        """

        self.taker_fee = self.config_input.get("taker_fee", 0.20)
        """taker_fee: float

        Taker fee percentage for the exchange, used to calculate fees in simulated trades.
        In live trading, actual fees from the exchange API are used.
        """

        # ------------------
        # Simulator Config
        # ------------------

        self.subset_size_days = self.config_input.get("subset_size_days", -1)
        """subset_size_days: int, days

        Size (in days) of subsets within the entire market candle history,
        used to partition the market and test the strategy on different conditions for robustness and generalizability:
        Increases the variance of simulated history data and lowers the risk of overfitting.
        If a strategy works well on most subsets, it should also work well on the whole market.
        -1 means no subsets, the whole market is used.
        """

        self._subset_size_increments = _hours_to_increments(24 * self.subset_size_days, self.candle_interval)
        """_subset_size_increments: int

        Conversion of subset_size_days to increments based on the candle_interval.
        Not intended for manual configuration.
        """

        self.subset_amount = self.config_input.get("subset_amount", 1)
        """subset_amount: int

        Number of subsets to create from the entire market history,
        with starting points evenly distributed across the history.
        Overlapping subsets are possible if:
        subset_amount * subset_size_days > fetch_init_timeframe_days.
        """

        self.train_val_split_ratio = self.config_input.get("train_val_split_ratio", 0.8)
        """train_val_split_ratio: float

        The whole market history can be split into a training and a validation set.
        The ratio defines the proportional size of the training set.
        Automatically the validation set will be the remaining part (1 - train_val_split_ratio).
        """

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
            "SPELL",
        ]
        """
        assets_excluded: list of symbols
        Assets that are stable coins or otherwise unsuitable for trading, to be excluded.
        """
        # ----------------------
        # Legacy config options
        # Maybe deprecated
        # ----------------------

        self.use_dbms = self.config_input.get("use_dbms", True)
        self.is_sim = self.config_input.get("dry_run", False)

    # ---------------
    # Config methods
    # ---------------

    def load_config(self, config_input: dict | None, config_file: str | None) -> dict:
        """Load the configuration from a provided input or a config.yaml file.

        First check if a configuration dictionary is provided. If not, try to load
        a configuration file from the specified search paths.
        If dict provided, flatten the configuration dictionary and return it.

        Params:
            config_input: An optional dictionary containing configuration data. If None, the function
                        tries to load a config.yaml file from the specified search paths.

        Returns:
            A flattened dictionary containing the loaded configuration data.

        Raises:
            SystemExit: If no configuration is provided and no configuration file is found in the search paths.
        """
        self.config_input = config_input if config_input else {}
        config_type = "dict"

        # Load config.yaml file if no config dict is provided
        if not self.config_input or config_file:
            config_paths.insert(0, str(config_file))
            self.config_input, config_type = _try_config_paths(config_paths)

        if not self.config_input:
            raise SystemExit(
                "No config file found. Please provide a config.yaml file or a config dict. "
                + f"Current search paths: {os.path.abspath('')}/config.yaml "
                + f"and {os.path.abspath('')}/config/config.yaml"
            )

        self.log.info("Loading config via %s", config_type)
        return _flatten_dict(self.config_input)

    def set_config_as_attr(self, config_input: dict) -> None:
        """Set all config values as attributes.

        Reverse of method to_dict().

        Params:
            config_input: A dict of config values.
        """
        # Set all config values as attributes. This ensures that user defined config keys are also accessible.
        for config_key, config_val in config_input.items():
            setattr(self, config_key, config_val)

    def to_dict(self, for_sim: bool = True) -> dict:
        """Return a dict of the Config object's attributes.

        If for_sim is True, exclude attributes that are
        not used in the simulator (and not convertable to float).

        Params:
            for_sim: Whether to exclude attributes that are not used in the simulator.

        Returns:
            A dict of the Config object's attributes.
        """
        attr_to_dict = self.__dict__

        if for_sim:
            attr_to_dict = {key: val for (key, val) in attr_to_dict.items() if isinstance(val, int | float | bool)}

        return attr_to_dict
