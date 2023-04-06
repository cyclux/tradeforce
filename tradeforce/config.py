"""Config module for tradeforce package.
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
    """Flatten nested config dict to one single level dict"""
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
    """Convert hours to increments corresponding to candle_interval.
    One increment is one candle timestep. e.g. 1h = 60min = 12 candles
    """
    candle_interval_min = candle_interval_to_min(candle_interval)
    return int(hours * 60 // candle_interval_min)


class Config:
    """The Config class is used to load and store the user config.
    It gets passed to all other classes and allows us to access them via self.config.<config_key>
    Provides default values to fall back on if param is not specified in user config.
    """

    # TODO: Add sanity check
    def __init__(self, root: Tradeforce, config_input: dict | None = None):
        self.log = root.logging.get_logger(__name__)

        config_input = config_input if config_input else {}
        config_type = "dict"

        # Load config.yaml file if no config dict is provided
        if not config_input:
            config_input, config_type = try_config_paths(config_paths)
        if not config_input:
            raise SystemExit(
                "No config file found. Please provide a config.yaml file or a config dict. "
                + f"Current search paths: {os.path.abspath('')}/config.yaml "
                + f"and {os.path.abspath('')}/config/config.yaml"
            )

        config_input = flatten_dict(config_input)

        self.log.info("Loading config via %s", config_type)

        # Set all config values as attributes. This ensures that user defined config keys are also accessible.
        for config_key, config_val in config_input.items():
            setattr(self, config_key, config_val)

        # If not specified, use default value.
        self.working_dir = Path(config_input.get("working_dir", os.path.abspath("")))
        self.run_live = config_input.get("run_live", False)
        self.update_mode = config_input.get("update_mode", "None").lower()
        self.log_level_live = config_input.get("log_level_live", "ERROR").upper()
        self.log_level_ws_update = config_input.get("log_level_ws_update", "ERROR").upper()

        self.exchange = config_input.get("exchange", "bitfinex")
        self.force_source = config_input.get("force_source", "none").lower()
        self.check_db_consistency = config_input.get("check_db_consistency", True)
        self.local_cache = config_input.get("local_cache", True)
        self.candle_interval = config_input.get("candle_interval", "5min")
        self.base_currency = config_input.get("base_currency", "USD")
        self.history_timeframe = config_input.get("history_timeframe", "120days")
        self.dbms = config_input.get("dbms", "mongodb").lower()
        self.dbms_host = config_input.get("dbms_host", "localhost")
        self.dbms_port = config_input.get("dbms_port", "1234")
        self.dbms_user = config_input.get("dbms_user", None)
        self.dbms_pw = config_input.get("dbms_pw", None)
        self.dbms_connect_db = config_input.get("dbms_connect_db", "postgres")
        self.dbms_db = config_input.get("dbms_db", f"{self.exchange}_db")
        self.dbms_history_entity_name = config_input.get("name", f"{self.exchange}_history_{self.history_timeframe}")

        self.creds_path = config_input.get("creds_path", self.working_dir)
        self.relevant_assets_cap = config_input.get("relevant_assets_cap", 100)

        self.trader_id = config_input.get("id", 1)
        self.moving_window_hours = config_input.get("moving_window_hours", 20)
        self.moving_window_increments = hours_to_increments(self.moving_window_hours, self.candle_interval)
        self.budget = float(config_input.get("budget", 0))
        self.buy_opportunity_factor = config_input.get("buy_opportunity_factor", 0.0)
        self.buy_opportunity_boundary = config_input.get("buy_opportunity_boundary", 0.02)
        self.profit_factor = config_input.get("profit_factor", 0.05)
        self.profit_ratio_limit = config_input.get("profit_ratio_limit", 1.01)
        self.prefer_performance = config_input.get("prefer_performance", 0)
        self.amount_invest_fiat = config_input.get("amount_invest_fiat", 0)
        # TODO: Implement max_buy_per_asset > 1
        self.max_buy_per_asset = config_input.get("max_buy_per_asset", 1)
        self.hold_time_limit = config_input.get("hold_time_limit", 10000)
        self.investment_cap = config_input.get("investment_cap", 0)
        self.maker_fee = config_input.get("maker_fee", 0.10)
        self.taker_fee = config_input.get("taker_fee", 0.20)
        self.use_dbms = config_input.get("use_dbms", True)
        self.is_sim = config_input.get("dry_run", False)
        # Simulator specific
        self.index_start = config_input.get("index_start", 0)
        self.snapshot_size = config_input.get("snapshot_size", -1)
        self.snapshot_amount = config_input.get("snapshot_amount", 1)
        self.sim_start_delta = config_input.get("sim_start_delta", None)
        self.sim_timeframe = config_input.get("sim_timeframe", None)

        # Following assets are either stable coins or not tradable without verification on bitfinex
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

    def to_dict(self, for_sim=True):
        """Return a dict of the Config object's attributes.
        If for_sim is True, exclude attributes that are not used in the simulator (and not convertable to float).
        """
        attr_to_dict = self.__dict__
        if for_sim:
            attr_to_dict = {key: val for (key, val) in attr_to_dict.items() if isinstance(val, int | float | bool)}
        return attr_to_dict
