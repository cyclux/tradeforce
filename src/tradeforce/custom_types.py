"""_summary_
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
