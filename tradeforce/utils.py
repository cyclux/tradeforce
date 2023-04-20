""" utils.py

Module: tradeforce.utils
------------------------

Helper functions for Tradeforce modules.

These functions facilitate various tasks such as converting time units,
extracting column names, converting symbols between exchange and standard formats,
calculating fees for buy and sell orders, and post-processing results from simulations.

Main functions in the module include:

    get_col_names: Extract unique column names from a given
        DataFrame or Series index.

    ms_to_ns, ns_to_ms, and ns_to_ms_array: Convert time units
        between milliseconds and nanoseconds.

    candle_interval_to_min: Convert candle interval to minutes.

    candle_interval_to_ms: Convert candle interval to milliseconds.

    convert_symbol_to_exchange and convert_symbol_from_exchange:
        Convert symbols between exchange and standard formats.

    get_timedelta: Get timedelta object and timestamp from a string.

    get_time_minus_delta: Get datetime and timestamp in UTC minus delta time.

    get_reference_index: Get a DataFrame datetime index with a given
        timeframe and frequency.

    calc_fee: Calculate fees for buy and sell orders.

    get_metric_labels, convert_sim_items, and get_sim_metrics_df:
        Functions useful for post-processing results from simulations.

"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

# import tradeforce.simulator.default_strategies as strategies
from tradeforce.custom_types import DictTimedelta, DictTimestamp

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.market.history import MarketHistory
    from tradeforce.config import Config


def get_col_names(idx: pd.Index, specific_col: str = "") -> list[str]:
    """Extract unique column names from a given Pandas DataFrame or Series index.

    Takes an index object from a DataFrame or Series and
    extracts unique column names by splitting the index labels based on an
    underscore delimiter. An optional specific column can be provided to
    filter the column names.

    Params:
        idx:          A Pandas Index object from a DataFrame or Series.
        specific_col: An optional string to filter the column names.
                      If provided, the filtered column names will include
                      the specific column with an underscore delimiter.

    Returns:
        A list of unique column names extracted from the given index.
    """

    return pd.unique(
        [f"{symbol.split('_')[0]}{'_' if specific_col else ''}{specific_col}" for symbol in list(idx)]
    ).tolist()


def ms_to_ns(t_ms: int) -> int:
    """Convert milliseconds to nanoseconds

    Params:
        t_ms: An integer representing a duration in milliseconds.

    Returns:
        An integer representing the equivalent duration in nanoseconds.
    """
    return int(t_ms * 10**6)


def ns_to_ms(t_ns: int) -> int:
    """Convert nanoseconds to milliseconds

    Params:
        t_ns: An integer representing a duration in nanoseconds.

    Returns:
        An integer representing the equivalent duration in milliseconds.
    """
    return int(t_ns // 10**6)


def ns_to_ms_array(t_ns: np.ndarray) -> np.ndarray:
    """Convert nanoseconds to milliseconds for an array of values

    Params:
        t_ns: Array representing a duration in nanoseconds.

    Returns:
        Array representing the equivalent duration in milliseconds.
    """
    return t_ns // 10**6


def candle_interval_to_min(candle_interval: str) -> int:
    """Convert candle interval to minutes. Quick mapping.

    Params:
        candle_interval: A string representing the candle interval. e.g. "5m" etc.

    Returns:
        An integer representing the candle interval in minutes.

    Raises:
        ValueError: If candle_interval is not supported.
    """
    candle_intervals = {
        "5min": 5,
        "15min": 15,
        "30min": 30,
        "1h": 60,
        "3h": 180,
        "6h": 360,
        "12h": 720,
        "1d": 1440,
        "3d": 4320,
        "1w": 10080,
    }

    if candle_interval not in candle_intervals.keys():
        raise ValueError(
            f"Candle interval {candle_interval} not supported. Choose from {list(candle_intervals.keys())}"
        )

    return candle_intervals[candle_interval]


def candle_interval_to_ms(candle_interval: str) -> int:
    """Convert candle interval to milliseconds.

    Convinience function to convert candle interval to milliseconds.

    Params:
        candle_interval: A string representing the candle interval. e.g. "5m" etc.

    Returns:
        An integer representing the candle interval in milliseconds.
    """
    return candle_interval_to_min(candle_interval) * 60 * 1000


def _convert_to_array(symbol_input: str | list | np.ndarray) -> np.ndarray:
    """Convert input to array and flatten if necessary

    Params:
        symbol_input: A string, list, or numpy array.

    Returns:
        A 1D numpy array.
    """
    return np.array([symbol_input]).flatten()


def convert_symbol_to_exchange(
    symbol_input: str | list | np.ndarray, base_currency: str = "USD", with_t_prefix: bool = True
) -> list[str]:
    """Convert symbol string to exchange format.

    "with_trade_prefix" is only relevant for bitfinex
    e.g. BTC -> tBTCUSD or META -> tMETA:USD

    Params:
        symbol_input:  A string, list, or numpy array.
        base_currency: A string representing the base currency.
        with_t_prefix: A boolean indicating whether to include the "t" prefix.

    Returns:
        A list of converted symbol strings.
    """
    symbol_input_normalized = _convert_to_array(symbol_input)
    t_prefix = "t" if with_t_prefix else ""
    symbol_output = [
        f'{t_prefix}{symbol}{":" if len(symbol) > 3 else ""}{base_currency}' for symbol in symbol_input_normalized
    ]
    return symbol_output


def convert_symbol_from_exchange(symbol_input: str | list | np.ndarray, base_currency: str = "USD") -> list[str]:
    """Convert symbol string from exchange format to standard format.

    Params:
        symbol_input:  A string, list, or numpy array.
        base_currency: A string representing the base currency.

    Returns:
        A list of converted symbol strings. e.g. ["tBTCUSD"] -> ["BTC"]

    """
    # Normalize input to array: so that 2D arrays and strings can be transformed as well
    symbol_input_normalized = _convert_to_array(symbol_input)

    # Filter for symbols which match the base_currency
    symbol_output = symbol_input_normalized[[symbol[-3:] == base_currency for symbol in symbol_input_normalized]]

    # Some symbols which have more than 3 characters (e.g. META:USD)
    # have a colon separating the symbol and base currency. Normalize (remove colon) for the next step.
    np_symbol_output = np.char.replace(symbol_output, f":{base_currency}", base_currency)

    # Remove base_currency and the "t" prefix
    np_symbol_output = np.char.replace(np_symbol_output, base_currency, "")
    np_symbol_output = np.char.replace(np_symbol_output, "t", "")

    return np_symbol_output.tolist()


def get_timedelta(delta: str | int, unit: Literal["ms", "min", "h"] | None = None) -> DictTimedelta:
    """Get timedelta object and timestamp from string.

    E.g. "1h" -> {"datetime": pd.Timedelta("1h"), "timestamp": 3600000}

    Params:
        delta: A string or integer representing a duration.
        unit:  A string representing the unit of the duration. e.g. "h", "m", "s", "ms"

    Returns:
        A custom typed dictionary containing a timedelta object and timestamp.
    """
    if unit:
        delta_datetime = pd.Timedelta(delta, unit=unit)
    else:
        delta_datetime = pd.Timedelta(delta)

    delta_timestamp = ns_to_ms(delta_datetime.value)

    return {"datetime": delta_datetime, "timestamp": delta_timestamp}


def get_now() -> DictTimestamp:
    """Get current datetime and timestamp in UTC.

    E.g. {"datetime": pd.Timestamp("2023-01-01 00:00:00"), "timestamp":1672531200000}

    Returns:
        A custom typed dictionary containing a timestamp object and timestamp.
    """
    now_datetime = pd.Timestamp.now(tz="UTC")
    now_timestamp = ns_to_ms(now_datetime.value)

    return {"datetime": now_datetime, "timestamp": now_timestamp}


def get_time_minus_delta(timestamp: int | None = None, delta: str = "") -> DictTimestamp:
    """Get datetime and timestamp in UTC minus delta time.

    Either from a given timestamp or the current time will be used.
    E.g. get_time_minus_delta(timestamp=1672531200000, delta="1h") ->
    {"datetime": pd.Timestamp("2023-01-01 00:00:00"), "timestamp":1672531200000}

    Params:
        timestamp: An integer representing a timestamp in milliseconds.
        delta:     A string representing a duration. e.g. "1min", "15min", "1h"

    Returns:
        A custom typed dictionary containing a timestamp object and timestamp.
    """
    if timestamp is not None:
        start_time: DictTimestamp = {
            "timestamp": timestamp,
            "datetime": pd.to_datetime(timestamp, unit="ms", utc=True),
        }
    else:
        start_time = get_now()

    # Convert delta to timedelta object and timestamp
    timedelta = get_timedelta(delta=delta)

    datetime = start_time["datetime"] - timedelta["datetime"]
    timestamp = start_time["timestamp"] - timedelta["timestamp"]

    return {
        "datetime": datetime,
        "timestamp": timestamp,
    }


def get_reference_index(timeframe: dict[str, pd.Timestamp], freq: str = "5min") -> pd.DataFrame:
    """Get a DataFrame datetime index
    -> with given timeframe and frequency.

    Can be utilized to create a ground truth to compare to
    and thus find differences (=missing data).

    Params:
        timeframe: Dict containing start and end datetime.
        freq:      A string representing the frequency of the index.
                    e.g. "5min", "15min", "1h"

    Returns:
        A datetime index as a DataFrame.
    """
    datetime_index = pd.date_range(
        start=timeframe["start_datetime"],
        end=timeframe["end_datetime"],
        freq=freq,
        name="t",
        inclusive="both",
    ).asi8

    df_datetime_index = pd.DataFrame(ns_to_ms_array(datetime_index), columns=["t"])
    return df_datetime_index


def calc_fee(config: Config, amount_asset: float, price_current: float, order_type: str) -> tuple[float, float, float]:
    """Calculate fees for buy and sell orders.

    'exchange_fee' is either taker or maker fee depending on order type.

    Params:
        config:        Contains exchange taker and maker fee.
        amount_asset:  A float representing the amount of the asset.
        price_current: A float representing the current price of the asset.
        order_type:    A string representing the order type -> ("buy" | "sell")

    Returns:
        A tuple containing the amount of the asset including fees,
        the amount of fees in asset currency and fiat currency.
    """
    # Normalize amount to positive value: Can be negative if order_type is "sell"
    amount_asset = abs(amount_asset)
    exchange_fee = config.taker_fee if order_type == "buy" else config.maker_fee
    amount_fee_asset = (amount_asset / 100) * exchange_fee
    amount_asset_incl_fee = amount_asset - amount_fee_asset
    amount_fee_fiat = np.round(amount_fee_asset * price_current, 3)
    return amount_asset_incl_fee, amount_fee_asset, amount_fee_fiat


# --------------------------------------------------------------------------#
# Following functions are useful to post-process results from simulations. #
# --------------------------------------------------------------------------#


def get_metric_labels() -> list[str]:
    """Mapping of metric labels to indices in simulation output array.

    Returns:
        List of metric labels.
    """
    metric_labels = [
        "asset_idx",
        "buy_factor",
        "row_index_buy",
        "price_buy",
        "price_incl_profit_factor_target",
        "amount_invest_per_asset",
        "amount_invest_asset",
        "amount_fee_buy_fiat",
        "budget",
        "row_index_sell",
        "price_sell",
        "amount_sold_fiat",
        "amount_sold_asset",
        "amount_fee_sell_fiat",
        "amount_profit_fiat",
        "value_asset_in_fiat",
        "total_value",
        "buy_orders",
    ]
    return metric_labels


def convert_sim_items(input_array: np.ndarray, asset_names: list) -> dict:
    """Convert simulation results from array to dictionary for easier access.

    Mapping of metric labels to indices in simulation output array.

    Params:
        input_array: Array containing simulation results.
        asset_names: A list of asset names.

    Returns:
        Dict containing simulation results.

    """
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


def get_sim_metrics_df(sim_trades_history: np.ndarray, market_history_instance: MarketHistory) -> pd.DataFrame:
    asset_names = market_history_instance.get_asset_symbols(updated=True)
    metric_labels = get_metric_labels()

    df_sim_trades_history = pd.DataFrame(sim_trades_history, columns=metric_labels)
    asset_idxs = df_sim_trades_history.loc[:, "asset_idx"].astype(int).to_list()

    df_sim_trades_history.insert(0, "asset", [asset_names[asset_idx] for asset_idx in asset_idxs])
    df_sim_trades_history["tt_sell"] = df_sim_trades_history["row_index_sell"] - df_sim_trades_history["row_index_buy"]

    df_sim_trades_history[
        ["asset_idx", "row_index_buy", "row_index_sell", "buy_orders", "tt_sell"]
    ] = df_sim_trades_history[["asset_idx", "row_index_buy", "row_index_sell", "buy_orders", "tt_sell"]].astype(int)

    return df_sim_trades_history


# def get_subset_indices(subset_idx_boundary, subset_amount=10, _subset_size_increments=10000):
#     subset_idx_boundary = subset_idx_boundary - _subset_size_increments
#     subset_idxs = np.linspace(0, subset_idx_boundary, subset_amount).astype(np.int64)
#     return subset_idxs
