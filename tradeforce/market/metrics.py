""" market/metrics.py

Module: tradeforce.metrics
--------------------------

Provides functions for analyzing the historical market data and calculating asset
performance metrics, such as trading volume, number of candles, candle sparsity,
and asset volatility. It also provides functions to calculate the buy performance
of each asset for live trading based on a moving window of historical data.


Imported Functions:

Main function:
    get_init_relevant_assets: Get the initial set of relevant assets
        based on the specified asset list and asset selection criteria.

    calc_asset_buy_performance: Calculate the buy performance of each asset
        in the market history based on moving window increments and an optional timestamp.

Helper functions:
    get_col_names: Extract and return column names from a DataFrame
        based on a specific pattern or column type.

    _nanstd_with_check: Calculate the standard deviation while ignoring NaN values,
        with an additional check for a single unique value.

    _calculate_start_end_idx_type: Calculate start and end indices and indexing type
        for selecting data from a DataFrame.

    _validate_and_calculate_buy_performance: Validate and calculate buy performance
        for each asset in the market window.

    candle_interval_to_ms: Convert a candle interval string to its equivalent duration in milliseconds.
    ms_to_candle_interval: Convert a duration in milliseconds to its equivalent candle interval string.
    _aggregate_history: Aggregate historical market data based on a specified timeframe.
    _validate_agg_timeframe: Validate the aggregation timeframe and raise an error if invalid.
    _get_volume_usd: Calculate the total trading volume in USD for each asset in a DataFrame.
    _get_amount_candles: Calculate the number of candles for each asset in a DataFrame.
    _calculate_candle_sparsity: Calculate the sparsity of candles for each asset in a DataFrame.
    _get_candle_sparsity: Calculate and return the candle sparsity for each asset in a DataFrame.
    _add_pct_change_cols: Add percentage change columns to a DataFrame containing asset history data.
    _get_asset_volatility: Calculate and return the volatility for each asset in a DataFrame.

"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Callable
from tradeforce.custom_types import DictRelevantAssets
from tradeforce.utils import get_col_names, candle_interval_to_ms

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import Tradeforce

# -------------------
# Prevent DoF errors
# -------------------


def _nanmax_with_check(input_values: list) -> float:
    """Calculate the maximum value of a list, accounting for NaN values.

    Params:
        input_values: List of numerical values.

    Returns:
        Maximum value in the list, ignoring NaN values. If the list
            contains only NaN values or is empty, return NaN.
    """
    non_nan_count = np.count_nonzero(~np.isnan(input_values))

    return (
        np.nanmax(input_values, initial=-np.inf)
        if non_nan_count > 1
        else input_values[0]
        if len(input_values) > 0
        else np.nan
    )


def _nanmin_with_check(input_values: list) -> float:
    """Calculate the minimum value of a list, accounting for NaN values.

    Params:
        input_values: List of numerical values.

    Returns:
        Minimum value in the list, ignoring NaN values. If the list
            contains only NaN values or is empty, return NaN.
    """

    non_nan_count = np.count_nonzero(~np.isnan(input_values))

    return (
        np.nanmin(input_values, initial=np.inf)
        if non_nan_count > 1
        else input_values[0]
        if len(input_values) > 0
        else np.nan
    )


def _nanstd_with_check(input_values: pd.DataFrame, ddof: int = 1) -> float:
    """Calculate the standard deviation, accounting for NaN values.

    Params:
        input_values: DataFrame containing numerical values.

        ddof:         Delta degrees of freedom (default is 1).
                        The divisor used in calculations is N - ddof,
                        where N represents the number of elements.

    Returns:
        Standard deviation of the DataFrame, ignoring NaN values. If the
            DataFrame contains only NaN values or is empty, return NaN.
    """
    non_nan_count = np.count_nonzero(~np.isnan(input_values))

    return float(
        np.nanstd(input_values, ddof=ddof)
        if non_nan_count > ddof
        else input_values.iloc[0]
        if len(input_values) > 0
        else np.nan
    )


# --------------------------
# Determine relevant assets
# --------------------------


async def get_init_relevant_assets(root: Tradeforce, capped: int = -1) -> DictRelevantAssets:
    """Analyze the market to identify relevant assets for trading.

    Fetch the initial market history, filter the assets based on
    configured criteria, and return a dictionary containing
    relevant assets, their metrics, and the market history data.

    Params:
        root:   Tradeforce intance, providing access to the
                    configuration, logging, and API objects.

        capped: Optional integer to limit the number of relevant assets returned.
                If capped > 0, the function will return only the top 'capped' assets
                sorted by the number of candles (default is -1, which returns all assets).

    Returns:
        Dict containing the following keys:
        - assets:  List of relevant asset symbols.
        - metrics: DataFrame with metrics for each relevant asset.
        - data:    Initial market history DataFrame.
    """
    root.log.info("Analyzing market for relevant assets...")

    init_timespan = _calculate_init_timespan(root)
    init_market_history = await root.market_updater_api.update_market_history(init_timespan=init_timespan)

    df_relevant_assets_metrics = _filter_relevant_assets(root, init_market_history)
    relevant_asset_symbols = df_relevant_assets_metrics.sort_values("amount_candles", ascending=False).index

    if capped > 0:
        relevant_asset_symbols = relevant_asset_symbols[:capped]

    root.log.info("Market analysis finished!")

    return {
        "assets": relevant_asset_symbols.tolist(),
        "metrics": df_relevant_assets_metrics,
        "data": init_market_history,
    }


# -------------------------------------------------
#  Helper functions for get_init_relevant_assets()
# -------------------------------------------------


def _get_asset_performance_metrics(df_input: pd.DataFrame, candle_interval: str) -> pd.DataFrame:
    """Compute performance metrics for assets based on historical candle data.

    Calculate the following metrics for each asset:
     - Volume in USD
     - Number of candles
     - Candle sparsity
     - Asset volatility

    Params:
        df_input:        DataFrame containing historical candle data
                            for multiple assets.

        candle_interval: String representing the time interval of the
                            candles (e.g. '5m', '1h', '1d').

    Returns:
        DataFrame containing the performance metrics for each asset, with columns:
        - vol_usd:         Volume in USD for each asset.
        - amount_candles:  The number of candles for each asset.
        - candle_sparsity: The candle sparsity for each asset.
        - volatility:      The volatility of each asset.
    """
    assets_volume_usd = _get_volume_usd(df_input)
    amount_candles = _get_amount_candles(df_input)
    candle_sparsity = _get_candle_sparsity(df_input)
    asset_volatility = _get_asset_volatility(df_input, candle_interval)

    asset_metrics = pd.DataFrame([assets_volume_usd, amount_candles, candle_sparsity, asset_volatility]).T
    asset_metrics.columns = pd.Index(["vol_usd", "amount_candles", "candle_sparsity", "volatility"])

    return asset_metrics


def _calculate_init_timespan(root: Tradeforce) -> str:
    """Calculate the initial timespan required to fetch market history data.

    The timespan is determined by the limit of the Bitfinex REST API for one request
    (10.000 candles) and the desired timeframe days defined in the configuration.

    Params:
        root: Tradeforce instance containing the configuration settings.

    Returns:
        String representing the initial timespan for fetching
            market history data, e.g. '34days'.
    """
    freq_for_max_request = pd.Timedelta(root.config.candle_interval) * 10_000

    return (
        str(freq_for_max_request.days) + "days"
        if root.config.fetch_init_timeframe_days >= 34
        else f"{root.config.fetch_init_timeframe_days}days"
    )


def _filter_relevant_assets(root: Tradeforce, init_market_history: pd.DataFrame) -> pd.DataFrame:
    """Filter relevant assets based on performance metrics and configuration settings.

    Assets are considered relevant if they meet the minimum number of candles and maximum
    candle sparsity criteria specified in the configuration settings.

    Params:
        root:                Tradeforce instance containing the configuration settings.
        init_market_history: DataFrame containing the initial market history data.

    Returns:
        DataFrame containing the relevant assets based on their performance metrics.
    """
    min_amount_candles = root.config.min_amount_candles
    max_candle_sparsity = root.config.max_candle_sparsity

    return _get_asset_performance_metrics(init_market_history, root.config.candle_interval).query(
        f"amount_candles >= {min_amount_candles} & candle_sparsity <= {max_candle_sparsity}"
    )


# ------------------------
# Calculate asset metrics
# ------------------------


def _apply_agg_func_to_columns(
    df_input: pd.DataFrame, candle_type_cols: list[str], agg_func: Callable, amount_intervals: int
) -> pd.DataFrame:
    """Apply the aggregation function to the columns of the input DataFrame
        as a rolling window based on the given candle type columns.

    Params:
        df_input:         DataFrame containing the input data.
        candle_type_cols: List of column names related to a specific
                            candle type (e.g., 'o', 'h', 'l', 'c', 'v').

        agg_func:         Callable aggregation function to apply
                            to the specified columns.

        amount_intervals: The number of intervals to consider for
                            the aggregation operation.

    Returns:
        DataFrame with the aggregation function applied to the specified columns.
    """
    forward_indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=amount_intervals)

    return (
        df_input.loc[:, candle_type_cols]
        .rolling(window=forward_indexer, step=amount_intervals, min_periods=1)
        .apply(agg_func, raw=True)
        .fillna(method="bfill")
    )


def _aggregate_history(df_input: pd.DataFrame, candle_interval: str, agg_timeframe: str = "1h") -> pd.DataFrame:
    """Aggregate market history data based on the desired aggregation timeframe.

    The function applies different aggregation functions
    to different types of candle data (e.g., open, high, low, close, volume).

    Params:
        df_input:        DataFrame containing the market history data.
        candle_interval: String representing the original candle interval (e.g., '5m').
        agg_timeframe:   String representing the desired aggregation timeframe (e.g., '1h').

    Returns:
        DataFrame containing the aggregated market history data.
    """
    amount_intervals = int(pd.Timedelta(agg_timeframe) / pd.Timedelta(candle_interval))

    agg_func_map: dict[str, Callable] = {
        "o": lambda row: row[0],
        "h": _nanmax_with_check,
        "l": _nanmin_with_check,
        "c": lambda row: row[-1],
        "v": np.nansum,
    }
    relevant_assets_agg_list = []

    for candle_type, func in agg_func_map.items():
        candle_type_cols = [col for col in df_input.columns if col[-1] == candle_type]
        relevant_assets_agg_list.append(_apply_agg_func_to_columns(df_input, candle_type_cols, func, amount_intervals))

    return pd.concat(relevant_assets_agg_list, axis=1)[df_input.columns]


def _get_volume_usd(df_input: pd.DataFrame) -> pd.Series:
    """Calculate the total volume in USD for each asset in the input DataFrame.

    Iterate through each asset, multiply its close price by its volume, and
    sum the results to obtain the total volume in USD for that asset.
    The calculated volumes are then sorted and returned as a Series.

    Params:
        df_input: DataFrame containing market history data with
                    columns for close and volume.

    Returns:
        Sorted Series containing the total volume in USD for each asset.
    """
    volume_usd = {}
    assets = get_col_names(df_input.columns)

    for asset in assets:
        volume_usd[asset] = int(np.sum(df_input[f"{asset}_c"] * df_input[f"{asset}_v"]))

    return pd.Series(volume_usd).sort_values()


def _get_amount_candles(df_input: pd.DataFrame) -> pd.Series:
    """Calculate the number of candles for each asset in the input DataFrame.

    Count the non-missing values in the volume columns for each asset,
    indicating the number of candles. Sort the results and return them as a Series.

    Params:
        df_input: DataFrame containing market history data with columns for volume.

    Returns:
        Sorted pandas Series containing the number of candles for each asset.
    """
    candles_vol = get_col_names(df_input.columns, specific_col="v")
    amount_candles = df_input[candles_vol].count()
    amount_candles.index = pd.Index(get_col_names(amount_candles.index))

    return amount_candles.sort_values()


def _calculate_candle_sparsity(df_vol: pd.DataFrame) -> dict[str, int]:
    """Calculate the sparsity of candles for each asset in the input DataFrame.

    Iterate through each volume column of the input DataFrame and calculate
    the mean difference between non-missing data points' indices,
    which represents the sparsity of candles for each asset.
    The calculated sparsity values are then returned as a dictionary.

    Params:
        df_vol: DataFrame containing volume data for each asset.

    Returns:
        Dict containing the calculated sparsity value for each asset.
    """
    df_candle_sparsity = {}

    for col in df_vol.columns:
        asset_index_not_nan = df_vol[col][pd.notna(df_vol[col])].index
        asset_index_not_nan /= 1000
        df_candle_sparsity[col] = int(np.diff(asset_index_not_nan).mean())

    return df_candle_sparsity


def _get_candle_sparsity(df_input: pd.DataFrame) -> pd.Series:
    """Calculate and return the candle sparsity for each asset in the input DataFrame.

    Params:
        df_input: DataFrame containing market history data with columns for volume.

    Returns:
        Series containing the calculated candle sparsity for each asset.
    """
    candles_vol = get_col_names(df_input.columns, specific_col="v")
    df_vol = df_input[candles_vol]

    df_candle_sparsity = _calculate_candle_sparsity(df_vol)

    series_candle_sparsity = pd.Series(df_candle_sparsity)
    series_candle_sparsity.index = pd.Index(get_col_names(series_candle_sparsity.index))

    return series_candle_sparsity


def _add_pct_change_cols(assets_history_asset: pd.DataFrame) -> pd.DataFrame:
    """Add percentage change columns to the input DataFrame.

    Params:
        assets_history_asset: DataFrame containing asset history data.

    Returns:
        DataFrame containing the percentage change columns for each original column.
    """
    cols_pct_change = {}
    for column in assets_history_asset.columns:
        cols_pct_change[f"{column}_pct"] = assets_history_asset.loc[:, column].pct_change() + 1

    return pd.DataFrame(cols_pct_change)


def _get_asset_volatility(df_input: pd.DataFrame, candle_interval: str) -> pd.Series:
    """Calculate and return the volatility for each asset in the input DataFrame.

    First extract the open price columns of the given DataFrame and aggregate
    them using the _aggregate_history function with a 1-hour aggregation timeframe.
    The percentage change of the aggregated open prices is then calculated using
    the _add_pct_change_cols function. Finally, compute the volatility of each asset
    as the standard deviation of the percentage changes.

    Params:
        df_input:        DataFrame containing market history data with
                            columns for open prices.

        candle_interval: String representing the candle interval used
                            in the market history data.

    Returns:
        Sorted Series containing the calculated volatility for each asset.
    """
    candles_open = get_col_names(df_input.columns, specific_col="o")
    assets_agg_open = _aggregate_history(df_input, candle_interval, agg_timeframe="1h").loc[:, candles_open]
    assets_agg_open_pct = _add_pct_change_cols(assets_agg_open)

    std_series = assets_agg_open_pct.apply(_nanstd_with_check, axis=0)
    return std_series.sort_values()


# -----------------------------------
# Asset performance for live trading
# -----------------------------------


def _calculate_start_end_idx_type(moving_window_increments: int, timestamp: int | None, candle_interval: str) -> tuple:
    """Calculate the start and end indices

    and the indexing type for selecting data from a DataFrame.

    Calculate the start and end indices for selecting data from a DataFrame based
    on the moving_window_increments and the provided timestamp.
    Also determine the indexing type ('iloc' or 'loc')
    based on whether a timestamp is provided or not.

    Params:
        moving_window_increments: Integer representing the number of moving
                                    window increments.

        timestamp:                Optional integer representing the timestamp
                                    to consider for the end index.

        candle_interval:          String representing the candle interval
                                    used in the market history data.

    Returns:
        Tuple containing the start index, end index, and indexing type.
    """
    start = -1 * moving_window_increments
    end = None
    idx_type = "iloc"

    if timestamp is not None:
        idx_type = "loc"
        candle_interval_in_ms = candle_interval_to_ms(candle_interval)
        start = timestamp - (moving_window_increments * candle_interval_in_ms)
        end = timestamp

    return start, end, idx_type


def _validate_and_calculate_price_performance(
    root: Tradeforce, market_window_pct_change: pd.DataFrame, moving_window_increments: int
) -> pd.Series:
    """Validate and calculate the price performance
    -> for each asset in the market window.

    Check if the amount of historical values is within the acceptable range for the
    given moving_window_increments. tollerance = 5 is arbitrarily chosen.
    Then calculate the "buy signal" represented by price performance:
    summing the percentage change of prices for each asset.

    Params:
        root:                     Tradeforce instance containing configuration
                                    and logging information.

        market_window_pct_change: DataFrame containing the market window
                                    percentage change data.

        moving_window_increments: Integer representing the number of moving
                                    window increments.

    Returns:
        Series containing the buy performance for each asset in the market window.
    """
    tollerance = 5
    warning_threshold_reached = len(market_window_pct_change) + tollerance < moving_window_increments

    if warning_threshold_reached:
        difference = moving_window_increments - len(market_window_pct_change)
        root.log.warning(
            "Missing %s candle entries to calculate the asset performance with set "
            + "'moving_window_increments'=%s Check DB consistency if the number of missing candles grows.",
            str(difference),
            str(moving_window_increments),
        )
    return market_window_pct_change.sum()


def calc_buy_signals(root: Tradeforce, moving_window_increments: int, timestamp: int | None = None) -> pd.Series:
    """Calculate the buy signals (price performance) of each asset in the market history.

    based on the provided moving window increments and an optional timestamp.
    Fetch the market history data, compute the percentage change, and validate
    the data before calculating the buy signals.

    Params:
        root:                     Tradeforce instance containing configuration
                                    and logging information.

        moving_window_increments: Integer representing the number of
                                    moving window increments.

        timestamp:                Optional integer representing the timestamp
                                    to consider for the end index.

    Returns:
        Series containing the buy performance for each asset in the market history.
    """
    start, end, idx_type = _calculate_start_end_idx_type(
        moving_window_increments, timestamp, root.config.candle_interval
    )
    market_window_pct_change = root.market_history.get_market_history(
        start=start,
        end=end,
        idx_type=idx_type,
        pct_change=True,
        pct_as_factor=False,
        metrics=["o"],
        fill_na=True,
        uniform_cols=True,
    )
    price_performance = _validate_and_calculate_price_performance(
        root, market_window_pct_change, moving_window_increments
    )
    return price_performance
