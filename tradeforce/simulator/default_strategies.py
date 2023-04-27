""" simulator/default_strategies.py

Module: tradeforce.simulator.default_strategies
-----------------------------------------------

Contains the default pre-processing, buy, and sell strategies for the
tradeforce trading simulation. Those default functions can be replaced by custom functions.
See examples/simulator_custom.py as an example.

Includes following functions:

    pre_process():        Prepare input data for the trading simulation by processing
                            historical market data and calculating buy signals.

    buy_strategy():       Execute the buy strategy by selecting assets that meet specific
                            conditions based on the buy signal scores and user-defined
                            preferences.

    sell_strategy():      Execute the sell strategy by identifying assets that meet specific
                            selling conditions based on whether the assets have reached their
                            target prices or if the hold time has elapsed and the current profit
                            ratio meets the "profit_factor_target_min".

    get_current_window(): Get the current moving window of price percentage changes from
                            the given historical price data. This is a potential helper
                            function for the buy strategy.

"""


from __future__ import annotations

import numpy as np
import numba as nb  # type: ignore

from typing import TYPE_CHECKING

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.config import Config
    from tradeforce.market.history import MarketHistory
    from pandas import DataFrame


# ---------------
# Pre-processing
# ---------------


def pre_process(
    config: Config, market_history: MarketHistory, dataset_type: str, train_val_split_idx: int
) -> dict[str, DataFrame]:
    """Prepare input data for the trading sim

    by processing historical market data and calculating "buy signal scores":

    Retrieve the asset prices and their percentage changes, apply clipping
    to the percentage changes to sanitize anomalies and compute the buy signal
    scores, using a rolling window sum.

    The dataset_type determines whether to retrieve the training or validation
    set, which is defined by the range "before" or "after" train_val_split_idx.

    If nor "train" or "val" is set as 'dataset_type': start and end are both None,
    which will return the entire dataset.

    This function is only called once per dataset_type before iterating over the
    respective dataset.

    Params:
        config:              Containing relevant settings for calculating
                                buy signal scores.

        market_history:      MarketHistory object containing historical market data.

        dataset_type:        Type of dataset to retrieve. Either "train" or "val".

        train_val_split_idx: Index of the training/validation split. Values before the
                                index are considered training data, values after
                                validation data.

    Returns:
        Dict containing the following keys and their corresponding DataFrames:
        - asset_prices:      Original asset prices.
        - asset_prices_pct:  Percentage changes in asset prices after clipping.
        - buy_signals:       Based on a rolling window sum of asset_prices_pct.
    """

    # None as start or end will default to the first or last index
    # e.g. start=None, end=1000 will return the first 1000 rows
    # and  start=1000, end=None will return the last 1000 rows
    start, end = None, None

    if dataset_type == "train":
        # start = None
        end = train_val_split_idx

    if dataset_type == "val":
        start = train_val_split_idx
        # end = None

    asset_prices = market_history.get_market_history(start=start, end=end, idx_type="iloc", metrics=["o"], fill_na=True)

    asset_prices_pct = market_history.get_market_history(
        start=start,
        end=end,
        idx_type="iloc",
        metrics=["o"],
        fill_na=True,
        pct_change=True,
        pct_as_factor=False,
        pct_first_row=0,
    )

    asset_prices_pct = _clip_asset_prices_pct(asset_prices_pct)
    buy_signals = _compute_buy_signals(config, asset_prices_pct)

    return {
        "asset_prices": asset_prices,
        "asset_prices_pct": asset_prices_pct,
        "buy_signals": buy_signals,
    }


def _clip_asset_prices_pct(asset_prices_pct: DataFrame) -> DataFrame:
    """Clip asset prices percentage changes
    -> between a lower and an upper threshold.

    This sanitizes anomalies in the market API data to realistic values.

    Clipping is done by computing the quantiles of the input and replacing
    values outside the specified thresholds with the quantile values.

    This approach is robust to the variance of the input and ensures
    only extreme outliers are clipped.

    Params:
        asset_prices_pct : DataFrame containing asset price
                            percentage changes.

    Returns:
        A DataFrame with the clipped asset price percentage changes.

    """
    lower_threshold = 0.000005
    upper_threshold = 0.999995
    quantiles = asset_prices_pct.stack().quantile([lower_threshold, upper_threshold])

    asset_prices_pct[asset_prices_pct > quantiles.loc[upper_threshold]] = quantiles.loc[upper_threshold]
    asset_prices_pct[asset_prices_pct < quantiles.loc[lower_threshold]] = quantiles.loc[lower_threshold]

    return asset_prices_pct


def _compute_buy_signals(config: Config, asset_prices_pct: DataFrame) -> DataFrame:
    """Compute the buy signal scores
    -> using a rolling window sum.

    Calculate the "buy signals" by applying a rolling window sum
    to the percentage changes in asset prices.

    The window size (moving_window_increments) is dependent on the 'candle_interval'
    and derived from 'moving_window_hours' in the configuration.

    min_periods is set to the window size to ensure all aggregated values are derived
    from the same amount of records (the window size). Within the first records (index
    smaller than the window size) the rolling window sum will return NaN values.
    To skip those NaN values, they are dropped -> [window - 1 :]


    Params:
        config:           Containing relevant settings for calculating
                                buy signal scores.

        asset_prices_pct: DataFrame containing the percentage change
                            in asset prices.

    Returns:
        DataFrame containing the computed buy signal scores.
    """
    window = int(config.moving_window_increments)
    return asset_prices_pct.rolling(window=window, step=1, min_periods=window).sum(engine="numba")[window - 1 :]


# --------------------------------------------------------------------------
# Default buy strategy
# --------------------------------------------------------------------------
# Recommended to pre-compute the buy_signals.
# However, it is also possible to compute it on the fly,
# this is much slower but more flexible to enable some use cases:
#
# window_history_prices_pct = get_current_window(params, asset_prices_pct)
# buyfactor_row = np.sum(window_history_prices_pct, axis=0)
# --------------------------------------------------------------------------


@nb.njit(cache=True, parallel=False)
def buy_strategy(params: dict, asset_prices_pct: np.ndarray, buy_signals: np.ndarray) -> np.ndarray:
    """Execute the buy strategy

    by selecting assets that meet specific conditions, based on user-defined
    preferences / given parameters: A buy signal.

    This function is called for each iteration step / candle record in the dataset.

    In this default implementation the "buy signal" was calculated by a moving
    window sum of the percentage changes in asset prices. This is a very simple
    approach to define a comparable score for asset price performance over a
    specific time period.

    If possible, it is recommended to pre-compute the buy signal scores for the
    whole dataset, as this is usually much faster than computing it "on the fly" on
    each iteration within this function. To get the corresponding buy signal
    score for the current iteration step, the pre-computed array ('buy_signals')
    can be indexed with the current row index ('row_idx' from the params dict).

    To determine the buy options / signals for the current iteration step, a boolean
    mask is created based on the user-defined parameters. The mask is then applied to
    select the relevant assets. Lastly, the signal scores are sorted based on the
    user-defined parameters. For details on this process see _get_buy_signal_mask()
    and _get_buy_option_array().

    Params:
        params: Dict containing parameters relevant for the buy strategy, including:

                - row_idx:                  Current row index of the iteration.
                - moving_window_increments: Size of the moving window in increments.
                - buy_signal_score:         Targeted buy signal score.
                - buy_signal_boundary:      Range around the buy_signal_score.
                - buy_signal_preference:    Preference for the score (e.g. sort for
                                                    highest or lowest)

        asset_prices_pct:  Array containing the percentage change in asset prices.

        buy_signals:       Array containing buy signal scores.

    Returns:
        Array containing the assets as indices, sorted based on their buy signal
            scores and user-defined preferences.
    """
    # The window size ('moving_window_increments') needs to be subtracted from the
    # 'row_idx' because we need the indices of both arrays to be aligned:
    # The 'buy_signals' array was shortened by the window size.
    # See _compute_buy_signals()
    row_idx = np.int64(params["row_idx"] - params["moving_window_increments"])
    buy_signal_scores = buy_signals[row_idx]

    buy_options_mask = _get_buy_signal_mask(params, buy_signal_scores)
    buy_options = _get_buy_signal_array(params, buy_signal_scores, buy_options_mask)

    return buy_options


@nb.njit(cache=True, parallel=False)
def _get_buy_signal_mask(params: dict, buy_signal_scores: np.ndarray) -> np.ndarray:
    """Generate a boolean mask

    representing assets that meet the specified buy signal score range:
    In this default implementation, the range is determined by the user-defined
    'buy_signal_boundary'. By subtracting and adding the boundary we get the
    min and max scores for the range. The mask is then created accordingly.

    Params:
        params: Dict containing the parameters for the buy strategy, including:
                - buy_signal_score:    Targeted buy signal score.
                - buy_signal_boundary: Range around the targeted buy_signal_score.

        buy_signal_scores: Array containing the buy signal scores.

    Returns:
        Boolean array (mask) where each element is True if the corresponding
            asset in the buy_signal_scores meets the buy_signal_boundary,
            and False otherwise.
    """
    buy_signal_score_min = params["buy_signal_score"] - params["buy_signal_boundary"]
    buy_signal_score_max = params["buy_signal_score"] + params["buy_signal_boundary"]
    return (buy_signal_scores >= buy_signal_score_min) & (buy_signal_scores <= buy_signal_score_max)


@nb.njit(cache=True, parallel=False)
def _get_buy_signal_array(params: dict, buy_signal_scores: np.ndarray, buy_signal_mask: np.ndarray) -> np.ndarray:
    """Generate an array containing the buy signals

    based on a price performance score and user-defined preferences:
    The buy options are represented as an array containing indices of assets that are
    within the buy signal score boundary / range. Finally the array is sorted
    based on 'buy_signal_preference'.

    Params:
        params: Dict containing the parameters for the buy strategy, including:
                - buy_signal_score: Targeted buy signal score.
                - buy_signal_boundary: Range around the targeted buy_signal_score.
                - buy_signal_preference: User preference for buying assets.

        buy_signal_scores:      Array containing the buy signal scores.

        buy_options_mask:       A boolean array (mask) representing assets
                                that meet the specified buy_signal_boundary.

    Returns:
        Array containing the assets as indices, sorted based on their buy signal
            scores and user-defined preferences.
    """
    # Need to be float as they get 'vstack'ed with float buy_signal_values
    buy_signal_indices = np.argwhere(buy_signal_mask).flatten().astype(np.float64)
    # The buy_option_values are used to sort the buy_option_indices
    buy_signal_values = buy_signal_scores[buy_signal_mask]

    # Prefer values closer to the target buy signal score
    if params["buy_signal_preference"] == 0:
        buy_signal_values = np.absolute(buy_signal_values - params["buy_signal_score"])

    # Prefer values with lower buy signal scores (ascending order)
    buy_option_array = np.vstack((buy_signal_indices, buy_signal_values))
    buy_option_array = buy_option_array[:, buy_option_array[1, :].argsort()]

    # Prefer values with higher buy signal scores (descending order -> flip axis)
    if params["buy_signal_preference"] == 1:
        buy_option_array = buy_option_array[:, ::-1]

    return buy_option_array.astype(np.int64)[0]


# ----------------------
# Default sell strategy
# ----------------------


@nb.njit(cache=True, parallel=False)
def sell_strategy(params: dict, buybag: np.ndarray, history_prices_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Execute the sell strategy

    by identifying assets that meet specific selling conditions based on the given parameters.
    So it determines whether assets in the buy bag can be sold.

    This function is called for each iteration step / candle record in the dataset.

    The default implementation checks for the following conditions:
    Whether the assets have reached their target prices or if the minimum sell conditions
    are met: hold time has elapsed and the current profit ratio meets the minimum profit.

    Params:
        params: Dict containing the parameters for the sell strategy, including:
                - "row_idx": Current row index.
                - "_hold_time_increments": Hold time before trying to sell an asset for the
                                            minimum profit factor target. Rerived from
                                            'hold_time_days' and 'candle_interval'.

                - "profit_factor_target_min": Minimum profit factor target for selling an
                                                asset after _hold_time_increments is reached.

        buybag:             Array containing the buy bag, which stores buy times, buy prices,
                                target prices etc.

        history_prices_row: Array containing the current prices of all assets in the market.

    Returns:
        A tuple containing two arrays:

        - Boolean array representing the sell decision for each asset in the buy bag.
            True indicates that the asset should be sold,
            and False indicates that the asset should be kept.
        - Array containing the current prices of the assets in the buy bag.
    """
    # Retrieve the indices and transform them,
    # so that they can be used to index the history_prices_row
    buy_option_indices = buybag[:, 0:1].T.flatten().astype(np.int64)
    current_prices = history_prices_row[buy_option_indices].reshape((1, -1)).T

    target_prices = buybag[:, 4:5]
    # Sanity check removes unrealistic market price changes and
    # thus prevents extreme biases due to exchange anomalies.
    # TODO: check performance in comparison to _clip_asset_prices_pct()
    current_prices = _apply_sanity_check(current_prices, target_prices)

    # The difference between the current iteration index and the
    # index at buy time determines the hold time in increments.
    increments_since_buy = params["row_idx"] - buybag[:, 2:3]
    # buybag[:, 3:4] is the buy price
    current_profit_ratios = current_prices / buybag[:, 3:4]

    is_target_profit_reached = current_prices >= target_prices
    is_min_sell_condition_reached = _check_min_sell_conditions(params, increments_since_buy, current_profit_ratios)

    sell_assets = (is_target_profit_reached | is_min_sell_condition_reached).flatten()

    return sell_assets, current_prices


@nb.njit(cache=True, parallel=False, fastmath=True)
def _apply_sanity_check(current_prices: np.ndarray, target_prices: np.ndarray) -> np.ndarray:
    """Apply sanity check on the current prices
    -> by comparing them to a threshold.

    If the ratio of the current price to the target price is greater than a
    specified threshold (1.2 in this case), the current price is replaced with
    the target price (capped). This helps prevent unrealistic and erroneous profits
    caused by anomalies in the exchange data.

    Params:
        current_prices: Array containing the current prices of the assets in the buy bag.
        target_prices:  Array containing the target prices of the assets in the buy bag.

    Returns:
        Array containing the updated current prices of the assets after applying the sanity check.
    """
    sanity_check_mask = (current_prices / target_prices > 1.2).flatten()
    current_prices[sanity_check_mask] = target_prices[sanity_check_mask]
    return current_prices


@nb.njit(cache=True, parallel=False)
def _check_min_sell_conditions(
    params: dict, times_since_buy: np.ndarray, current_profit_ratios: np.ndarray
) -> np.ndarray:
    """Calculate the sell conditions

    based on hold time and the current profit ratios of the assets in the buy bag:

    Using the given parameter (profit_factor_target_min), this function determines
    if an asset should be sold based on the hold time (_hold_time_increments) and the
    current profit ratios. Note that if profit_factor_target_min < 1 then the asset
    will probably be sold at a loss.

    The value of _hold_time_increments is derived from the hold_time_days defined in
    the config and is depending on candle_interval.

    It checks if the time since the buy is greater the hold time (max_hold_time_reached)
    and if the current profit ratio is greater or equal to the minimum target profit
    factor (min_profit_factor_reached).

    Params:
        params:                Dict containing the parameters for the sell strategy, including:
                               - "_hold_time_increments": The number of increments for the hold time.
                               - "profit_factor_target_min": The minimum target profit factor.
        times_since_buy:       Array containing the time since each asset in the buy bag was bought.
        current_profit_ratios: Array containing the current profit ratios of the assets in the buy bag.

    Returns:
        A boolean array where each element is True if the corresponding asset in the buy bag
        meets the hold time and profit ratio conditions for selling, and False otherwise.
    """
    max_hold_time_reached = times_since_buy > params["_hold_time_increments"]
    min_profit_factor_reached = current_profit_ratios >= params["profit_factor_target_min"]

    return max_hold_time_reached & min_profit_factor_reached


# ---------------------------------------------------
# Potential helper function for buy strategy
# For reference see "Default buy strategy" docstring
# ---------------------------------------------------


@nb.njit(cache=True, parallel=False)
def get_current_window(params: dict, history_prices_pct: np.ndarray) -> np.ndarray:
    """Retrieve the current window of the moving window.

    'row_idx' represents the current iteration index of the simulation.
    The moving window is useful to aggregate data from the past. So the
    start index is calculated by subtracting the size of the moving window.

    Params:
        params: Dict containing the parameters for the buy strategy, including:
                - "row_idx":                  Current row index.
                - "moving_window_increments": Size of the moving window in increments.

        history_prices_pct: Array containing the history of price percentage change of
                                all assets in the market.

    Returns:
        DataFrame containing the current window of the moving window.
    """
    moving_window_start = int(params["row_idx"] - params["moving_window_increments"])
    moving_window_end = int(params["row_idx"])

    return history_prices_pct[moving_window_start:moving_window_end]
