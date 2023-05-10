"""


Config module containing the Config class for Tradeforce.

Loads config dict or config.yaml file and stores it in a Config class
which can get passed to all other modules, classes and functions.

The Config class handles various settings and configurations for Tradeforce,
including data handling settings, exchange settings, trader and strategy
settings, and simulator configurations. It is responsible for loading, updating,
and storing these settings, as well as providing a method to represent
the configuration in a dictionary format.
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

    Args:
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

    Args:
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

    Args:
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
    """Config object is used to load and store the user config.

    It gets passed to all other classes and allows us to access them via self.config.<config_key>
    Provides default values to fall back on if param is not specified in user config.

    Args:
        root: The main Tradeforce instance.
                Provides access to logger or any other module.
        config_input: Dictionary containing the configuration.
        config_file: Absolute path to the configuration file (yaml).
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
        """str: Path to the working directory.

        Default is the directory Tradeforce is executed from.
        """

        self.check_db_consistency = self.config_input.get("check_db_consistency", True)
        """bool: Determines whether to check the in-memory database consistency

        by verifying no missing candles between the start and end timestamps.
        Useful to disable in hyperparameter search scenarios for performance.

        Default: ``True``
        """

        self.check_db_sync = self.config_input.get("check_db_sync", True)
        """bool: Checks whether in-memory database is in sync

        with the backend database. If not, it will try to sync bi-directionally by
        exchanging missing candles between the backend and in-memory database.
        During simulation mode, check_db_sync is usually not necessary and can be
        disabled for performance improvements (decreases load times).

        Default: ``True``
        """

        # ---------------
        # Logging Levels
        # ---------------

        self.log_level_ws_live = self.config_input.get("log_level_ws_live", "ERROR").upper()
        """str: Log level for the private websocket connection during live trading.

        Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

        Default: ``ERROR`` (only errors and critical messages are logged)
        """

        self.log_level_ws_update = self.config_input.get("log_level_ws_update", "ERROR").upper()
        """str: Log level for the public websocket connection (candle/price updater).

        Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

        Default: ``ERROR`` (only errors and critical messages are logged)
        """

        # -------------------------
        # Market history & Updates
        # -------------------------

        self.exchange = self.config_input.get("exchange", "bitfinex").lower()
        """str: Name of the exchange to use.

        Note:
            Currently, only "bitfinex" is supported.

        Default: ``bitfinex``
        """

        self.base_currency = self.config_input.get("base_currency", "USD")
        """str: Base currency to trade against, such as "USD" or "BTC".

        Default: ``USD``
        """

        self.candle_interval = self.config_input.get("candle_interval", "5min")
        """str: Interval for update candles.

        Shorter intervals result in more candles and increased precision,
        while longer intervals are more efficient but less precise.

        Options: "1min", "5min", "15min", "30min", "1h", "3h", "6h", "12h", "1D", "1W"

        Default: ``5min``
        """

        self.fetch_init_timeframe_days = self.config_input.get("fetch_init_timeframe_days", 120)
        """int: Number of days to load the candle history from.

        Note:
            Timeframe is calculated from the current time backwards.

            E.g. 120 days -> 120 days ago until now.

        Default: ``120``
        """

        self.force_source = self.config_input.get("force_source", "none").lower()
        """str: Specifies a particular data source for the candle history.

        Options:
            - ``none``: Search in order: local_cache, backend, api.
            - ``local_cache``: ".arrow" dump of in-memory DB (:py:attr:`working_dir`/data).
            - ``backend``: Fetch history from DB.
            - ``api``: Fetch history from exchange via Websocket.

        Default: ``none``
        """

        self.update_mode = self.config_input.get("update_mode", "none").lower()
        """str: Determines method to update market candle data.

        ``none``: Do not update.
                    Just load data of present DB / cache.
                    Useful for simulation mode or analysis.

        ``once``: Only update once
                    via REST API. Useful for cases where there
                    is already data present (DB or local cache)
                    and we just want to update it before starting
                    simulations.

        ``live``: Update continuously
                    via Websocket. Useful for live trading or
                    running a dedicated market server.

        Default: ``none``
        """

        # ---------------------
        # Database Connection
        # ---------------------

        self.dbms = self.config_input.get("dbms", "postgres").lower()
        """str: Determines which "Database Management System" (DBMS) to use.

        Note:
            Currently, Postgres and MongoDB are supported. Choose Postgres for
            full functionality, including hyperparameter search via Optuna.

        Default: ``postgres``
        """

        # TODO: Reference docker-compose.yml
        self.dbms_host = self.config_input.get("dbms_host", "docker_postgres")
        """str: Fully qualified domain name (FQDN)
        of the Database Management System (DBMS)
        such as ``localhost`` or a remote address.

        Note:
            The default host ``docker_postgres`` is defined by docker-compose.yml
            in the examples folder.

        Default: ``docker_postgres``
        """

        # TODO: Reference docker-compose.yml
        self.dbms_port = int(self.config_input.get("dbms_port", 5433))
        """int: Port number of the DBMS.

        The default port ``5433`` is defined by docker-compose.yml
        in the examples folder.

        Note:
            Port 5433 is used instead of the default port 5432 of Postgres
            to avoid conflict with a potential present Postgres installation.

        Default: ``5433``
        """

        # TODO: Reference docker-compose.yml
        self.dbms_user = self.config_input.get("dbms_user", "postgres")
        """str: Username of :py:attr:`dbms_db`.

        Default user is defined by docker-compose.yml in the examples folder.

        Default: ``postgres``
        """

        # TODO: Reference docker-compose.yml
        self.dbms_pw = self.config_input.get("dbms_pw", "postgres")
        """str: Password for :py:attr:`dbms_db`.

        Default password is defined by docker-compose.yml in the examples folder.

        Default: ``postgres``
        """

        self.dbms_connect_db = self.config_input.get("dbms_connect_db", "postgres")
        """str: Name of an existing database
        to connect to when creating a new database.

        Note:
            Specific for SQL DBMS like Postgres:
            To create a new database, a connection to an existing database with
            appropriate permissions is needed.

        Default: ``postgres``
        """

        self.dbms_db = self.config_input.get("dbms_db", f"{self.exchange}_db")
        """str: Name of the database to use
        for storing candle history and trader specific data.

        Note:
            Gets created if it does not exist.

        Note:
            Optuna specific data including results of "Studies"
            is stored in a separate database "optuna".

        Default gets generated from :py:attr:`exchange` _db

        E.g. ``bitfinex_db``
        """

        self.name = self.config_input.get("name", f"{self.exchange}_history_{self.fetch_init_timeframe_days}days")
        """str: Name of the "entity" which stores the candle history data.

        In the example configs declared within the ``market_history`` category.

        Note:
            Also determines the name of the :py:attr:`local_cache` file (.arrow in data folder).

        Note:
            "Entity" is used as a unifying term to describe
            a TABLE in Postgres or COLLECTION in MongoDB.

        Default gets generated from:
        :py:attr:`exchange` _history_ :py:attr:`fetch_init_timeframe_days` days

        E.g. ``bitfinex_history_120days``
        """

        self._dbms_history_entity_name = self.name
        """For internal use: convert the generic "name" to more specific name """

        self.local_cache = self.config_input.get("local_cache", True)
        """bool: Whether to save the in-memory database
        to a local ``.arrow`` file for faster loading in subsequent runs.

        Note:
            Gets saved on every candle update (:py:attr:`candle_interval`).

        Note:
            Gets saved into :py:attr:`working_dir`/data.

        Default: ``True``
        """

        # -------------------
        # Exchange Settings
        # -------------------

        self.credentials_path = self.config_input.get("credentials_path", self.working_dir)
        """str: Absolute path to the credentials file
        for authentication to the private Bitfinex API (for live trading).

        Note:
            This credentials file is ``SHA3_256`` encrypted by a user defined password.
            See :py:attr:`run_live` for more information.

        Default: :py:attr:`working_dir`
        """

        self.run_live = self.config_input.get("run_live", False)
        """bool: Whether to run Tradeforce in live trading mode
        executing the configured strategies on the exchange.

        Note:
            Needs valid credentials in :py:attr:`credentials_path`:
            If :py:attr:`run_live` is ``True`` Tradeforce will ask for the API key and
            secret to the exchange. On every live run the credentials will be decrypt after
            providing the correct password.

        CAUTION:
            This is NOT a simulation!
            The strategy is executed on the exchange.
            Real money can be lost!

        Default: ``False``
        """

        # ----------------------------
        # Trader & Strategy Settings
        # ----------------------------

        self.trader_id = int(self.config_input.get("id", 1))
        """int: Unique identifier for a trader instance.
        Useful when running multiple live trading strategies on the same database.

        Default: ``1``
        """

        self.budget = float(self.config_input.get("budget", 0))
        """float: Initial trading budget in :py:attr:`base_currency` (e.g. USD)

        Only used for simulations. This value sets the initial budget balance.

        Note:
            In live trading (:py:attr:`run_live` -> ``True``) this setting will be ignored / overwritten. The budget
            balance will be dynamically derived from the wallet of the exchange.

        Default: ``0``
        """

        self.amount_invest_per_asset = self.config_input.get("amount_invest_per_asset", 0.0)
        """float: Amount of :py:attr:`base_currency` (e.g., USD) to invest
        in each asset upon purchase.

        Default: ``0.0``
        """

        # Constraints for relevant assets

        self.relevant_assets_cap = int(self.config_input.get("relevant_assets_cap", 100))
        """int: Maximum number of relevant assets to load initially.

        If no data is present yet: either from the database or :py:attr:`local_cache`,
        a new dataset is fetched from the exchange. This setting limits the number of assets to fetch.

        Note:
            This cap is applied after filtering for relevant asset constraints:
            See :py:attr:`relevant_assets_min_amount_candles` and
            :py:attr:`relevant_assets_max_candle_sparsity`.

        Default: ``100``
        """

        self.relevant_assets_min_amount_candles = int(self.config_input.get("relevant_assets_min_amount_candles", 2000))
        """int: Minimum number of candles required
        for an asset to be considered relevant.

        Note:
            "Relevant assets" get filtered while initially fetching the candle history via API.

        Default: ``2000``
        """

        # TODO: Explain calculation -> _calculate_candle_sparsity
        self.relevant_assets_max_candle_sparsity = int(
            self.config_input.get("relevant_assets_max_candle_sparsity", 500)
        )
        """int: Maximum "sparcity" constraint of candles
        to be considered as relevant asset.

        Note:
            "Relevant assets" get filtered while initially fetching the candle history via API.

        Default: ``500``
        """

        # TODO: link to buy_strategy in docs
        self.moving_window_hours = self.config_input.get("moving_window_hours", 20)
        """int: Timeframe (in hours)
        of the moving window, to calculate the :py:attr:`buy_signal_score`
        which is utilized in the buy_strategy

        Default: ``20``
        """

        # TODO: Convert to private attribute?
        self.moving_window_increments = _hours_to_increments(self.moving_window_hours, self.candle_interval)
        """int: Number of candle increments in the moving window.

        Derived from :py:attr:`candle_interval` and :py:attr:`moving_window_hours`.

        CAUTION:
            Not intended for manual configuration.
            Use :py:attr:`moving_window_hours` setting instead.

        Default: Calculated by ``_hours_to_increments()``
        """

        # Default trading strategy parameters

        # TODO: Explain buy_signal_score somewhere
        self.buy_signal_score = self.config_input.get("buy_signal_score", 0.0)
        """float: Target buy signal score
        to consider an asset for purchase at a given point in time.

        Specifically, this point in time is the current ``params["row_idx"]``
        processed by :py:attr:`tradeforce.simulator.default_strategies.buy_strategy`.

        Note:
            Default implementation to calculate the ``buy_signal_score`` is the
            "sum of percentage changes" within the moving window
            range (:py:attr:`moving_window_hours`)
            See :py:attr:`tradeforce.simulator.default_strategies._compute_buy_signals`.
        """

        self.buy_signal_boundary = self.config_input.get("buy_signal_boundary", 0.02)
        """float: Target lower and upper limit of the :py:attr:`buy_signal_score`
        defining a range of acceptable scores for purchasing assets
        at a given point in time.

        Specifically, this point in time is the current ``params["row_idx"]``
        processed by :py:attr:`tradeforce.simulator.default_strategies.buy_strategy`.

        e.g. ``buy_signal_boundary == 0.02`` means +/-0.02 :py:attr:`buy_signal_score`
        to be considered for purchase.
        """

        self.buy_signal_preference = self.config_input.get("buy_signal_preference", 1)
        """int: Target preference for buying assets
        with :py:attr:`buy_signal_score` values:

            - ``1`` decending  -> higher values prefered,
            - ``-1`` ascending  -> lower values prefered,
            - ``0`` closest -> to the :py:attr:`buy_signal_score`.

        The assets with their respective :py:attr:`buy_signal_score` are chosen to
        be purchased in the order from left to right (lower index to higher index).

        Note:
            The ``buy_signal_preference`` determines the order of values within
            the :py:attr:`buy_signal_boundary`. Thus it determines which asset
            is prefered to be purchased at a given point in time.

        Default: ``1``
        """

        self.profit_factor_target = self.config_input.get("profit_factor_target", 0.05)
        """float: Target profit factor.

        This value determines the targeted profit on each trade.

        e.g. ``profit_factor_target == 1.05`` represents a target of 5% profit on each trade.
        So if an asset would be purchased for 100 USD, it would be sold for 105 USD
        once the price is reached.

        Default: ``0.05``
        """

        self.profit_factor_target_min = self.config_input.get("profit_factor_target_min", 1.01)
        """float: Minimum profit factor once :py:attr:`hold_time_days` is reached,
        allowing the trader to sell assets at lower profits or even losses
        (``profit_factor_target_min < 1``). This is useful to avoid holding assets
        for too long or to cut losses.

        Note:
            This setting is only relevant if :py:attr:`hold_time_days` is set ``> 0``.

        Default: ``1.01``
        """

        self.hold_time_days = self.config_input.get("hold_time_days", 4)
        """int: Number of days to hold an asset before reducing
        the :py:attr:`profit_factor_target` to :py:attr:`profit_factor_target_min`.

        Note:
            Internally :py:attr:`_hold_time_increments` (based on
            :py:attr:`candle_interval` and ``hold_time_days``) is primarily used for any
            calculation or processing. One ``increment`` represents one data record / candle time stamp.
            (every record is one :py:attr:`candle_interval` appart from the next)

        Default: ``4``
        """

        self._hold_time_increments = _hours_to_increments(24 * self.hold_time_days, self.candle_interval)
        """_hold_time_increments: int

        Conversion of hold_time_days to increments based on the candle_interval,
        used as an internal time reference.
        """

        # TODO: implement max_buy_per_asset > 1
        self.max_buy_per_asset = self.config_input.get("max_buy_per_asset", 1)
        """int: Maximum number of times a single asset can be bought.
        Useful for managing repeated purchases of an asset with a dropping price.

        Note:
            Currently only ``1`` supported.

        Default: ``1``
        """

        # TODO: implement investment_cap for live trader

        self.investment_cap = self.config_input.get("investment_cap", 0.0)
        """float: Maximum amount of budget (in :py:attr:`base_currency` e.g. USD)
        that can be invested in the market simultaneously:
        Useful for limiting reinvestment of profits.

        Note:
            Currently only implemented for simulations.
            Live trader ignores this setting:
            All potential profits are reinvested.

        Note:
            ``0.0`` equals unlimited investment, no cap.
            All potential profits are reinvested.

        Default: ``0.0``
        """

        self.maker_fee = self.config_input.get("maker_fee", 0.10)
        """float: Maker fee percentage
        used to calculate exchange maker fee in simulated trades.

        Maker fee is the fee paid to the exchange for adding liquidity to the market.
        Usually lower than taker fee and paid for buy orders.
        (which are not instantly filled).

        Note:
            In live trading, actual maker fee from exchange API is applied.

        Default: ``0.10``
        """

        self.taker_fee = self.config_input.get("taker_fee", 0.20)
        """float: Taker fee percentage
        used to calculate exchange taker fee in simulated trades.

        Taker fee is the fee paid to the exchange for removing liquidity from the market.
        Usually higher than maker fee and paid for market sell orders.

        Note:
            In live trading, actual taker fee from exchange API is applied.

        Default: ``0.20``
        """

        # ------------------
        # Simulator Config
        # ------------------

        self.subset_size_days = self.config_input.get("subset_size_days", -1)
        """ int: Size (in days) of subsets
        within the entire market candle history. Used to partition the market history
        data and test the strategy on different (start) conditions for robustness and
        generalizability.

        Note:
            Partitioning into subsets increases the variance of simulated history data
            and lowers the risk of overfitting. In other words: if a strategy works
            well on most subsets, it should also work better on the whole market.

        Note:
            ``-1`` equals no subsets, the whole market is used.

        Default: ``-1``
        """

        self._subset_size_increments = _hours_to_increments(24 * self.subset_size_days, self.candle_interval)
        """int: Conversion of :py:attr:`subset_size_days` to increments
        based on the :py:attr:`candle_interval`.

        CAUTION:
            Not intended for manual configuration. Internal use only.
        """

        self.subset_amount = self.config_input.get("subset_amount", 1)
        """int: Number of subsets to create
        (from the entire market history).
        Subset starting points (indices) are evenly distributed across the history.

        Note:
            Overlapping subsets are possible if:
            :py:attr:`subset_amount` * :py:attr:`subset_size_days`
            > :py:attr:`fetch_init_timeframe_days`.

        Default: ``1``
        """

        self.train_val_split_ratio = self.config_input.get("train_val_split_ratio", 0.8)
        """float: Ratio of training data (compared to validation data)

        The whole market history can be split into a training and a validation set.
        The ratio defines the proportional size of the training set to the whole market history.

        Note:
            Automatically the validation set will be the remaining part
            (1 - :py:attr:`train_val_split_ratio`). So in case of ``train_val_split_ratio = 0.8``
            the training data would be 80%, the validation data ``0.2`` (20%) of the whole
            market history.

        Default: ``0.8``
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

        Default: ``["UDC", "UST", "EUT", "EUS", "MIM", "PAX", "DAI", "TSD", "TERRAUST",
        "LEO", "WBT", "RBT", "ETHW", "SHIB", "ETH2X", "SPELL"]``
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

        Args:
            config_input: An optional dictionary containing configuration data. If None, the function
                        tries to load a config.yaml file from the specified search paths.

        Returns:
            A flattened dictionary containing the loaded configuration data.

        Raises:
            SystemExit: If no configuration is provided and no configuration file is found in the search paths.
        """
        self.config_input = config_input if config_input else {}

        # Config type is either "dict" or "path to yaml file"
        config_type = "dict"

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

        Args:
            config_input: A dict of config values.
        """
        # Set all config values as attributes. This ensures that user defined config keys are also accessible.
        for config_key, config_val in config_input.items():
            setattr(self, config_key, config_val)

    def to_dict(self, for_sim: bool = True) -> dict:
        """Return a dict of the Config object's attributes.

        If for_sim is True, exclude attributes that are
        not used in the simulator (and not convertable to float).

        Args:
            for_sim: Whether to exclude attributes that are not used in the simulator.

        Returns:
            A dict of the Config object's attributes.
        """
        attr_to_dict = self.__dict__

        if for_sim:
            attr_to_dict = {key: val for (key, val) in attr_to_dict.items() if isinstance(val, int | float | bool)}

        return attr_to_dict
