"""
Config module for tradeforce package.
Loads config dict or config.yaml file and stores it in a Config class
which can get passed to all other modules, classes and functions.
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


def load_yaml_config(config_path) -> dict:
    """Load config.yaml file"""
    try:
        with open(config_path, "r", encoding="utf8") as stream:
            yaml_config = yaml.safe_load(stream)
    except FileNotFoundError:
        return {}
    return dict(yaml_config)


def try_config_paths(config_paths: list) -> tuple[dict, str]:
    while len(config_paths) > 0:
        current_config_path = config_paths.pop(0)
        congig_path_absolute = os.path.join(os.path.abspath(""), current_config_path)
        config_input = load_yaml_config(congig_path_absolute)
        if config_input:
            break
    return config_input, congig_path_absolute


def flatten_dict(input_config_dict: dict) -> dict:
    """
    Flatten nested config dict to one single level dict
    """
    output_dict = {}

    def flatten(input_dict, key=""):
        if isinstance(input_dict, dict):
            for key in input_dict:
                flatten(input_dict[key], key)
        else:
            output_dict[key] = input_dict

    flatten(input_config_dict)
    return dict(output_dict)


def hours_to_increments(hours: int, candle_interval: str) -> int:
    """
    Convert hours to increments corresponding to candle_interval.
    One increment is one candle timestep. e.g. 1h = 60min = 12 candles
    """
    candle_interval_min = candle_interval_to_min(candle_interval)
    return int(hours * 60 // candle_interval_min)


class Config:
    """
    The Config class is used to load and store the user config.
    It gets passed to all other classes and allows us to access them via self.config.<config_key>
    Provides default values to fall back on if param is not specified in user config.
    """

    def __init__(self, root: Tradeforce, config_input: dict | None = None):
        self.log = root.logging.get_logger(__name__)
        self.config_input: dict = self.load_config(config_input)

        # Set all config keys as attributes of the Config class
        # This allows us to access them via self.config.<config_key> in any module
        self.set_config_as_attr(self.config_input)

        ##################
        # Core setup #
        ##################

        """
        working_dir: Path to working directory
        The path to the working directory. Default is the directory where Tradeforce is executed.
        """
        self.working_dir = Path(self.config_input.get("working_dir", os.path.abspath("")))

        """
        check_db_consistency: True, False

        Determines whether to check the in-memory database consistency by verifying no missing candles between
        the start and end timestamps. Useful to disable in hyperparameter search scenarios for performance.
        """
        self.check_db_consistency = self.config_input.get("check_db_consistency", True)

        ##################
        # Logging Levels
        ##################

        """
        log_level_ws_live: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        Log level for the private websocket connection during live trading.

        """
        self.log_level_ws_live = self.config_input.get("log_level_ws_live", "ERROR").upper()

        """
        log_level_ws_update: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        Log level for the public websocket connection (candle/price updater).
        """
        self.log_level_ws_update = self.config_input.get("log_level_ws_update", "ERROR").upper()

        ######################
        # Market history & Updates
        ######################

        """
        exchange: "bitfinex"
        The name of the exchange to use. Currently, only Bitfinex is supported.
        """
        self.exchange = self.config_input.get("exchange", "bitfinex")

        """
        base_currency: "USD"
        The base currency to trade against, such as "USD" or "BTC".
        """
        self.base_currency = self.config_input.get("base_currency", "USD")

        """
        candle_interval: str: "1min", "5min", "15min", "30min", "1h", "3h", "6h", "12h", "1D", "1W"

        The interval for market history and update candles.
        Shorter intervals result in more candles and less precision,
        while longer intervals are more efficient but less precise.
        """
        self.candle_interval = self.config_input.get("candle_interval", "5min")

        """
        history_timeframe_days: currently needs to be in "days".
        The number of days in the past to load the candle history from.
        """
        self.history_timeframe_days = self.config_input.get("history_timeframe_days", 120)

        """
        force_source: "none", "local_cache", "backend", "api"

        Specifies a particular data source for the candle history:
        "none" Search in order: local_cache, backend, api.
        "local_cache": "arrow" dump of in-memory DB (working_dir/data).
        "backend": Fetch history from DB.
        "api": Fetch history from exchange via Websocket.
        """
        self.force_source = self.config_input.get("force_source", "none").lower()

        """
        update_mode: "none", "once", "live"

        Update candle history of the DB to the current time.
        none: Do not update
        once: Only update statically once via REST API
        live: Update continuously via Websocket

        Determines how to update the candle history of the database:
        "none": Do not update.
        "once": Static update via REST API.
        "live": Continuous update via Websocket.
        """
        self.update_mode = self.config_input.get("update_mode", "none").lower()

        #######################
        # Database Connection
        #######################

        """
        dbms : "postgres", "mongodb"

        The "DataBase Management System" to use.
        Currently, Postgres and MongoDB are supported.
        Choose Postgres for full functionality, including hyperparameter search via optuna.
        """
        self.dbms = self.config_input.get("dbms", "postgres").lower()

        """
        dbms_host: str

        The fully qualified domain name (FQDN) of the database management system,
        such as "localhost" or a remote address. "docker_postgres" is the default for the docker container.
        """
        self.dbms_host = self.config_input.get("dbms_host", "docker_postgres")

        """
        dbms_port: int
        The port number of the DBMS.
        """
        self.dbms_port = int(self.config_input.get("dbms_port", 5432))

        """
        dbms_user: str
        Username of the dbms_db. "postgres" is the default user of the postgres docker container.
        """
        self.dbms_user = self.config_input.get("dbms_user", "postgres")

        """
        dbms_pw: str
        The password for the database. "postgres" is the default password of the postgres docker container.
        """
        self.dbms_pw = self.config_input.get("dbms_pw", "postgres")

        """
        dbms_connect_db: "postgres"

        The name of an existing database to connect to when creating a new database:
        Specific to SQL DBMS like Postgres.
        """
        self.dbms_connect_db = self.config_input.get("dbms_connect_db", "postgres")

        """
        dbms_db: str
        The name of the database to use. Gets created if it does not exist.
        """
        self.dbms_db = self.config_input.get("dbms_db", f"{self.exchange}_db")

        """
        dbms_history_entity_name: str
        The name of the entity (TABLE in Postgres or COLLECTION in MongoDB) to store candle history data.
        """
        self.dbms_history_entity_name = self.config_input.get(
            "name", f"{self.exchange}_history_{self.history_timeframe_days}days"
        )

        """
        local_cache: bool

        Whether to save the in-memory database to a local arrow file for faster loading in subsequent runs.
        Gets saved in working_dir/data.
        """
        self.local_cache = self.config_input.get("local_cache", True)

        #####################
        # Exchange Settings
        #####################

        """
        credentials_path: str

        The path to the exchange credentials file for authenticating the private websocket connection (Live trader).
        """
        self.credentials_path = self.config_input.get("credentials_path", self.working_dir)

        """
        run_live: bool

        Whether to run Tradeforce in live trading mode, executing the configured strategy on the exchange.
        Needs valid credentials in credentials_path.
        CAUTION: This is NOT a backtest / simulation. The strategy is executed on the exchange.
        """
        self.run_live = self.config_input.get("run_live", False)

        ##############################
        # Trader & Strategy Settings #
        ##############################

        """
        trader_id: int
        Unique identifier for a trader instance, useful when running multiple trading strategies on the same database.
        """
        self.trader_id = int(self.config_input.get("id", 1))

        """
        budget: float

        Initial trading budget in base currency (e.g., USD) for the trader, used primarily for simulations.
        In live trading, the budget is derived from the base currency balance in the exchange wallet.
        """
        self.budget = float(self.config_input.get("budget", 0))

        """
        amount_invest_per_asset: float
        Amount of base currency (usually fiat, e.g., USD) to invest in each asset upon purchase.
        """
        self.amount_invest_per_asset = self.config_input.get("amount_invest_per_asset", 0.0)

        """
        relevant_assets_cap: int
        Maximum number of assets to consider for trading or simulations, applied after filtering for relevant assets.
        """
        self.relevant_assets_cap = int(self.config_input.get("relevant_assets_cap", 100))

        """
        moving_window_hours: int

        Timeframe (in hours) for the moving window used in the simulation strategy,
        affecting the aggregation of various metrics for each asset.
        e.g. the mean or any other metric, like buy_performance_score.
        """
        self.moving_window_hours = self.config_input.get("moving_window_hours", 20)

        """
        _moving_window_increments: int

        Number of candle increments in the moving window, derived from candle_interval and moving_window_hours.
        Not intended for manual configuration.
        """
        self._moving_window_increments = hours_to_increments(self.moving_window_hours, self.candle_interval)

        # Default trading strategy parameters
        """
        buy_performance_score: float

        Performance score of an asset within the moving window.
        In the default strategy implementation typically calculated
        as the "sum of percentage changes" within the window.
        """
        self.buy_performance_score = self.config_input.get("buy_performance_score", 0.0)

        """
        buy_performance_boundary: float

        Lower and upper limit for the buy_performance_score,
        defining a range of acceptable scores for purchasing assets.
        e.g. 0.02 means the buy_performance_score must be +/-0.01 buy_performance_score to be considered for purchase.
        """
        self.buy_performance_boundary = self.config_input.get("buy_performance_boundary", 0.02)

        """
        buy_performance_preference: -1, 0, 1

        Preference for buying assets with performance
        positive (1) -> decending, higher values prefered,
        negative (-1) -> ascending, lower values prefered,
        or closest to the buy_performance_score (0).
        """
        self.buy_performance_preference = self.config_input.get("buy_performance_preference", 1)

        """
        profit_factor_target: float, positive

        Target profit factor for each asset.
        e.g. 1.05 represents a target of 5% profit on each trade.
        """
        self.profit_factor_target = self.config_input.get("profit_factor_target", 0.05)

        """
        profit_factor_target_min: float

        Minimum profit factor after _hold_time_increments,
        allowing the trader to sell assets at lower profits or even losses.
        """
        self.profit_factor_target_min = self.config_input.get("profit_factor_target_min", 1.01)

        """
        hold_time_days: int
        Number of days to hold an asset before reducing the profit_factor to profit_factor_min.
        """
        self.hold_time_days = self.config_input.get("hold_time_days", 4)

        """
        _hold_time_increments: int
        Conversion of hold_time_days to increments based on the candle_interval, used as an internal time reference.
        """
        self._hold_time_increments = hours_to_increments(24 * self.hold_time_days, self.candle_interval)

        # TODO: implement max_buy_per_asset > 1
        """
        max_buy_per_asset: int -> currently only 1 supported

        Maximum number of times a single asset can be bought,
        useful for managing repeated purchases of an asset with a dropping price.
        """
        self.max_buy_per_asset = self.config_input.get("max_buy_per_asset", 1)

        # TODO: implement investment_cap for live trader
        """
        investment_cap: float, currently only implemented for simulations

        Maximum amount of budget (in base currency) that can be invested in the market simultaneously,
        useful for limiting reinvestment of profits.

        """
        self.investment_cap = self.config_input.get("investment_cap", 0.0)

        """
        maker_fee: float

        Maker fee percentage for the exchange, used to calculate fees in simulated trades.
        In live trading, actual fees from the exchange API are used.
        """
        self.maker_fee = self.config_input.get("maker_fee", 0.10)

        """
        taker_fee: float

        Taker fee percentage for the exchange, used to calculate fees in simulated trades.
        In live trading, actual fees from the exchange API are used.
        """
        self.taker_fee = self.config_input.get("taker_fee", 0.20)

        ####################
        # Simulator Config #
        ####################

        """
        subset_size_days: int, days

        Size (in days) of subsets within the entire market candle history,
        used to partition the market and test the strategy on different conditions for robustness and generalizability:
        Increases the variance of simulated history data and lowers the risk of overfitting.
        If a strategy works well on most subsets, it should also work well on the whole market.
        -1 means no subsets, the whole market is used.
        """
        self.subset_size_days = self.config_input.get("subset_size_days", -1)

        """
        _subset_size_increments: int

        Conversion of subset_size_days to increments based on the candle_interval.
        Not intended for manual configuration.
        """
        self._subset_size_increments = hours_to_increments(24 * self.subset_size_days, self.candle_interval)

        """
        subset_amount: int

        Number of subsets to create from the entire market history,
        with starting points evenly distributed across the history.
        Overlapping subsets are possible if:
        subset_amount * subset_size_days > history_timeframe_days.
        """
        self.subset_amount = self.config_input.get("subset_amount", 1)

        """
        assets_excluded: list of symbols
        Assets that are stable coins or otherwise unsuitable for trading, to be excluded.
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

        #########################
        # Legacy config options #
        # Maybe deprecated      #
        #########################

        self.use_dbms = self.config_input.get("use_dbms", True)
        self.is_sim = self.config_input.get("dry_run", False)

    def load_config(self, config_input):
        self.config_input = config_input if config_input else {}
        config_type = "dict"

        # Load config.yaml file if no config dict is provided
        if not self.config_input:
            self.config_input, config_type = try_config_paths(config_paths)
        if not self.config_input:
            raise SystemExit(
                "No config file found. Please provide a config.yaml file or a config dict. "
                + f"Current search paths: {os.path.abspath('')}/config.yaml "
                + f"and {os.path.abspath('')}/config/config.yaml"
            )

        self.log.info("Loading config via %s", config_type)
        return flatten_dict(self.config_input)

    def set_config_as_attr(self, config_input: dict):
        # Set all config values as attributes. This ensures that user defined config keys are also accessible.
        for config_key, config_val in config_input.items():
            setattr(self, config_key, config_val)

    def to_dict(self, for_sim=True):
        """Return a dict of the Config object's attributes.
        If for_sim is True, exclude attributes that are not used in the simulator (and not convertable to float).
        """
        attr_to_dict = self.__dict__
        if for_sim:
            attr_to_dict = {key: val for (key, val) in attr_to_dict.items() if isinstance(val, int | float | bool)}
        return attr_to_dict
