"""Helper functions for Cryptobot

"""

from __future__ import annotations
from configparser import ConfigParser
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from typing_extensions import TypedDict
from bfxapi import Client  # type: ignore
from bfxapi.constants import WS_HOST, PUB_WS_HOST, REST_HOST, PUB_REST_HOST  # type: ignore
import tradeforce.simulator.default_strategies as strategies

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.config import Config
    from tradeforce import TradingEngine

TimestampDict = TypedDict("TimestampDict", {"datetime": pd.Timestamp, "timestamp": int})
TimedeltaDict = TypedDict("TimedeltaDict", {"datetime": pd.Timedelta, "timestamp": int})


def get_col_names(idx: pd.Index, specific_col: str = "") -> list[str]:
    """Get column names from Index of a DataFrame or Series"""
    return pd.unique(
        [f"{symbol.split('_')[0]}{'_' if specific_col else ''}{specific_col}" for symbol in list(idx)]
    ).tolist()


def ms_to_ns(t_ms: int):
    """Convert milliseconds to nanoseconds"""
    return int(t_ms * 10**6)


def ns_to_ms(t_ns: int) -> int:
    """Convert nanoseconds to milliseconds"""
    return int(t_ns // 10**6)


def ns_to_ms_array(t_ns: np.ndarray) -> np.ndarray:
    """Convert nanoseconds to milliseconds"""
    return t_ns // 10**6


def convert_to_array(symbol_input):
    """Convert input (str | list) to array"""
    return np.array([symbol_input]).flatten()


def convert_symbol_to_exchange(
    symbol_input: str | list | np.ndarray, base_currency="USD", with_t_prefix=True
) -> list[str]:
    """Convert symbol string to exchange format.
    "with_trade_prefix" is only relevant for bitfinex e.g. BTCUSD -> tBTCUSD"""
    symbol_input_normalized = convert_to_array(symbol_input)
    t_prefix = "t" if with_t_prefix else ""
    symbol_output = [
        f'{t_prefix}{symbol}{":" if len(symbol) > 3 else ""}{base_currency}' for symbol in symbol_input_normalized
    ]
    return symbol_output


def convert_symbol_from_exchange(symbol_input: str | list | np.ndarray, base_currency="USD") -> list[str]:
    """Convert symbol string from exchange format to standard format."""
    symbol_input_normalized = convert_to_array(symbol_input)
    symbol_output = symbol_input_normalized[[symbol[-3:] == base_currency for symbol in symbol_input_normalized]]
    np_symbol_output = np.char.replace(symbol_output, f":{base_currency}", base_currency)
    np_symbol_output = np.char.replace(np_symbol_output, base_currency, "")
    np_symbol_output = np.char.replace(np_symbol_output, "t", "")
    return np_symbol_output.tolist()


def get_timedelta(delta: str = "", unit=None) -> TimedeltaDict:
    """Get timedelta object and timestamp from string.
    E.g. "1h" -> {"datetime": pd.Timedelta("1h"), "timestamp": 3600000}
    unit: "s", "ms", "us", "ns" or "D", "h", "m", "s", "ms", "us", "ns"
    """
    delta_datetime = pd.Timedelta(delta, unit=unit)
    delta_timestamp = ns_to_ms(delta_datetime.value)
    return {"datetime": delta_datetime, "timestamp": delta_timestamp}


def get_now() -> TimestampDict:
    """Get current datetime and timestamp in UTC.
    E.g. {"datetime": pd.Timestamp("2023-01-01 00:00:00"), "timestamp":1672531200000}
    """
    now_datetime = pd.Timestamp.now(tz="UTC")
    now_timestamp = ns_to_ms(now_datetime.value)
    return {"datetime": now_datetime, "timestamp": now_timestamp}


def get_time_minus_delta(timestamp: int | None = None, delta="") -> TimestampDict:
    """Get datetime and timestamp in UTC minus delta time.
    E.g. get_time_minus_delta(timestamp=1672531200000, delta="1h") ->
    {"datetime": pd.Timestamp("2023-01-01 00:00:00"), "timestamp":1672531200000}
    """
    if timestamp is not None:
        start_time: TimestampDict = {
            "timestamp": timestamp,
            "datetime": pd.to_datetime(timestamp, unit="ms", utc=True),
        }
    else:
        start_time = get_now()
    timedelta = get_timedelta(delta=delta)

    datetime = start_time["datetime"] - timedelta["datetime"]
    timestamp = start_time["timestamp"] - timedelta["timestamp"]

    return {
        "datetime": datetime,
        "timestamp": timestamp,
    }


def get_reference_index(timeframe: dict[str, pd.Timestamp], freq="5min") -> pd.DataFrame:
    """Get a DataFrame datetime index with given timeframe and frequency.
    Can be utilized to create a ground truth to compare to and thus find differences (+missing data)."""
    datetime_index = pd.date_range(
        start=timeframe["start_datetime"],
        end=timeframe["end_datetime"],
        freq=freq,
        name="t",
        inclusive="both",
    ).asi8
    df_datetime_index = pd.DataFrame(ns_to_ms_array(datetime_index), columns=["t"])
    return df_datetime_index


def load_credentials(creds_path) -> dict[str, str] | None:
    """Load API credentials from credential config file.
    Returns None if file is not found or credentials are not valid.
    """
    creds_store = ConfigParser()
    credentials = {}
    try:
        creds_store.read(creds_path)
        credentials["auth_key"] = creds_store["api_cred"]["auth_key"]
        credentials["auth_sec"] = creds_store["api_cred"]["auth_sec"]
    except (TypeError, KeyError):
        return None
    return credentials


def connect_api(config: Config, api_type=None) -> Client | None:
    """Connect to Bitfinex API.
    Returns None if credentials are not valid.
    """
    credentials = load_credentials(config.creds_path)
    bfx_api = None
    if credentials is not None and api_type == "priv":
        bfx_api = Client(
            credentials["auth_key"],
            credentials["auth_sec"],
            ws_host=WS_HOST,
            rest_host=REST_HOST,
            logLevel=config.log_level_live,
        )
    if api_type == "pub":
        bfx_api = Client(
            ws_host=PUB_WS_HOST,
            rest_host=PUB_REST_HOST,
            ws_capacity=25,
            max_retries=100,
            logLevel=config.log_level_ws_update,
        )

    return bfx_api


def calc_fee(config, volume_crypto, price_current, order_type):
    """Calculate fees for buy and sell orders.
    exchange_fee is either taker or maker fee depending on order type."""
    volume_crypto = abs(volume_crypto)
    exchange_fee = config.taker_fee if order_type == "buy" else config.maker_fee
    amount_fee_crypto = (volume_crypto / 100) * exchange_fee
    volume_crypto_incl_fee = volume_crypto - amount_fee_crypto
    amount_fee_fiat = np.round(amount_fee_crypto * price_current, 3)
    return volume_crypto_incl_fee, amount_fee_crypto, amount_fee_fiat


def monkey_patch(root: TradingEngine, buy_strategy, sell_strategy) -> None:
    """Monkey patch user defined buy and sell strategies if provided"""
    if buy_strategy is not None:
        root.log.info("Custom buy_strategy loaded")
        # nb.njit(cache=False)(buy_strategy)
        strategies.buy_strategy = buy_strategy
    else:
        root.log.info("Default buy_strategy loaded")

    if sell_strategy is not None:
        root.log.info("Custom sell_strategy loaded")
        # nb.njit(cache=False)(sell_strategy)
        strategies.sell_strategy = sell_strategy
    else:
        root.log.info("Default buy_strategy loaded")


def candle_interval_to_min(candle_interval: str) -> int:
    """Convert candle interval to minutes"""
    candle_intervals = {
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "3h": 180,
        "6h": 360,
        "12h": 720,
        "1d": 1440,
        "3d": 4320,
        "1w": 10080,
    }
    return candle_intervals.get(candle_interval, 5)


def drop_dict_na_values(record, dbms):
    """Drop all values from dict that are NaN"""
    # In postgres insert_many operations require same length for all entries
    if dbms == "postgresql":
        return record
        # return {key: record[key] for key in record if not pd.isna(record[key])}
    return {key: record[key] for key in record if not pd.isna(record[key])}


# TODO: Following functions are not currently used. Check relevance


def get_filtered_from_nan(payload_insert) -> dict:
    for entry in payload_insert:
        payload_insert = {items[0]: items[1] for items in entry.items() if pd.notna(items[1])}
    return payload_insert


def get_metric_labels():
    metric_labels = [
        "asset_idx",
        "buy_factor",
        "row_index_buy",
        "price_buy",
        "price_incl_profit_factor",
        "amount_invest_fiat",
        "amount_invest_crypto",
        "amount_fee_buy_fiat",
        "budget",
        "row_index_sell",
        "price_sell",
        "amount_sold_fiat",
        "amount_sold_crypto",
        "amount_fee_sell_fiat",
        "amount_profit_fiat",
        "value_crypto_in_fiat",
        "total_value",
        "buy_orders",
    ]
    return metric_labels


def convert_sim_items(input_array, asset_names):
    """Converts simulation items to dictionary for easier access"""
    metric_labels = get_metric_labels()
    array_map = {
        metric_labels[0]: asset_names[int(input_array[0])],
        metric_labels[1]: np.round(input_array[1], 3),
        metric_labels[2]: int(input_array[2]),
        metric_labels[3]: input_array[3],
        metric_labels[4]: np.round(input_array[4], 10),
        metric_labels[5]: input_array[5],
        metric_labels[6]: input_array[6],
        metric_labels[7]: np.round(input_array[7]),
        metric_labels[8]: np.round(input_array[8], 2),
        metric_labels[9]: int(input_array[9]),
        metric_labels[10]: input_array[10],
        metric_labels[11]: input_array[11],
        metric_labels[12]: input_array[12],
        metric_labels[13]: input_array[13],
        metric_labels[14]: np.round(input_array[14], 2),
        metric_labels[15]: np.round(input_array[15], 2),
        metric_labels[16]: np.round(input_array[16], 2),
        metric_labels[17]: int(input_array[17]),
    }
    return array_map


def get_sim_metrics_df(sim_trades_history_array, market_history_instance):
    metric_labels = get_metric_labels()
    sim_trades_history_df = pd.DataFrame(sim_trades_history_array, columns=metric_labels)
    asset_names = market_history_instance.get_asset_symbols(updated=True)
    asset_idxs = sim_trades_history_df.loc[:, "asset_idx"].astype(int).to_list()
    sim_trades_history_df.insert(0, "asset", [asset_names[asset_idx] for asset_idx in asset_idxs])
    sim_trades_history_df["tt_sell"] = sim_trades_history_df["row_index_sell"] - sim_trades_history_df["row_index_buy"]

    sim_trades_history_df[
        ["asset_idx", "row_index_buy", "row_index_sell", "buy_orders", "tt_sell"]
    ] = sim_trades_history_df[["asset_idx", "row_index_buy", "row_index_sell", "buy_orders", "tt_sell"]].astype(int)
    return sim_trades_history_df


# def get_snapshot_indices(snapshot_idx_boundary, snapshot_amount=10, snapshot_size=10000):
#     snapshot_idx_boundary = snapshot_idx_boundary - snapshot_size
#     snapshot_idxs = np.linspace(0, snapshot_idx_boundary, snapshot_amount).astype(np.int64)
#     return snapshot_idxs
