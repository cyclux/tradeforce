""" custom_types.py

Module: tradeforce.custom_types
-------------------------------

Custom types for type hinting.

Defines various TypedDict classes for type hinting.
The TypedDict classes in this module include:

    DictCandle: Represents the structure of a single candlestick data point,
        including timestamp, open, close, high, low, volume, and symbol.

    DictRelevantAssets: Contains information about a list of relevant assets,
        their associated metrics, and their data in a DataFrame.

    DictTimestamp: Represents both a pandas Timestamp
        and its equivalent integer timestamp.

    DictTimedelta: Represents both a pandas Timedelta
        and its equivalent integer timestamp.

    DictTimeframe: Contains start and end timestamps (integer)
        and their equivalent pandas Timestamps.

    DictTimeframeExtended: Contains a DictTimeframe object and an additional
        field for the remaining time until a wait period is over (in milliseconds).

    DictMarketHistoryUpdate: A dictionary mapping asset symbols to a list of dictionaries,
        where each dictionary contains market history updates with keys representing
        either a timestamp (integer) or a value (float).

"""


from typing import TypedDict
import pandas as pd


# Type hinting for candle data
DictCandle = TypedDict(
    "DictCandle",
    {
        "mts": int,
        "open": float,
        "close": float,
        "high": float,
        "low": float,
        "volume": float,
        "symbol": str,
    },
)


DictRelevantAssets = TypedDict(
    "DictRelevantAssets",
    {
        "assets": list[str],
        "metrics": pd.DataFrame,
        "data": pd.DataFrame,
    },
)

DictTimestamp = TypedDict("DictTimestamp", {"datetime": pd.Timestamp, "timestamp": int})
DictTimedelta = TypedDict("DictTimedelta", {"datetime": pd.Timedelta, "timestamp": int})


DictTimeframe = TypedDict(
    "DictTimeframe",
    {
        "start_timestamp": int,
        "start_datetime": pd.Timestamp,
        "end_timestamp": int,
        "end_datetime": pd.Timestamp,
    },
)

DictTimeframeExtended = TypedDict(
    "DictTimeframeExtended",
    {"timeframe": DictTimeframe, "ms_until_wait_over": int},
)


DictMarketHistoryUpdate = dict[str, list[dict[str, int | float]]]
