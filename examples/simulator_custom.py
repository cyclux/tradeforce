import numpy as np
import numba as nb  # type: ignore
import pandas as pd

from tradeforce import Tradeforce
from tradeforce.market.history import MarketHistory
from tradeforce.config import Config

CONFIG = {
    "trader": {
        "budget": 10000,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
            "amount_invest_per_asset": 100,
            "investment_cap": 0,
            "buy_performance_score": 0.10,
            "buy_performance_boundary": 0.05,
            "buy_performance_preference": 1,
            "hold_time_days": 4,
            "profit_factor_target": 1.10,
            "profit_factor_target_min": 1.01,
            "moving_window_hours": 180,
        },
    },
    "backend": {
        "dbms": "postgresql",
        "dbms_host": "docker_postgres",
        "dbms_port": 5433,
        "dbms_connect_db": "postgres",
        "dbms_user": "postgres",
        "dbms_pw": "postgres",
        "local_cache": False,
    },
    "market_history": {
        "name": "bitfinex_history_2y",
        "exchange": "bitfinex",
        "base_currency": "USD",
        "candle_interval": "5min",
        "fetch_init_timeframe_days": 60,
        "update_mode": "none",
        "force_source": "local_cache",
        "check_db_sync": False,
    },
    "simulation": {
        "subset_size_days": 30,
        "subset_amount": 10,
    },
}


"""
Pre-processing
"""


def pre_process(config: Config, market_history: MarketHistory) -> dict[str, pd.DataFrame]:
    """
    Prepares input data for the trading sim by processing historical market data and calculating asset performance.

    The pre_process function takes a configuration object and a market history object as input.
    It retrieves the asset prices and their percentage changes,
    applies clipping to the percentage changes to sanitize anomalies, and computes the asset
    market performance using a rolling window sum.

    Params:
        config: Containing relevant settings for calculating asset market performance.
        market_history: A MarketHistory object containing historical market data.

    Returns:
        Dict containing the following keys and their corresponding DataFrames:
        - "asset_prices": Original asset prices.
        - "asset_prices_pct": Percentage changes in asset prices after clipping.
        - "asset_performance": Asset market performance based on a rolling window sum of asset_prices_pct.
    """
    asset_prices = market_history.get_market_history(metrics=["o"], fill_na=True)
    asset_prices_pct = market_history.get_market_history(
        metrics=["o"], fill_na=True, pct_change=True, pct_as_factor=False, pct_first_row=0
    )
    asset_prices_pct = _clip_asset_prices_pct(asset_prices_pct)
    asset_market_performance = _compute_asset_performance(config, asset_prices_pct)

    return {
        "asset_prices": asset_prices,
        "asset_prices_pct": asset_prices_pct,
        "asset_performance": asset_market_performance,
    }


def _clip_asset_prices_pct(asset_prices_pct: pd.DataFrame) -> pd.DataFrame:
    """
    Clips asset prices percentage changes between a lower and an upper threshold.
    Sanitizes anomalies in the market API data to realistic values.

    Takes a DataFrame containing asset price percentage changes and clips
    the values between the specified lower and upper thresholds. Clipping is done by
    computing the quantiles of the data and replacing values outside the specified
    thresholds with the quantile values.

    Params:
        asset_prices_pct : A DataFrame containing asset price percentage changes.

    Returns:
        A DataFrame with the clipped asset price percentage changes.

    """
    lower_threshold = 0.000005
    upper_threshold = 0.999995
    quantiles = asset_prices_pct.stack().quantile([lower_threshold, upper_threshold])

    asset_prices_pct[asset_prices_pct > quantiles.loc[upper_threshold]] = quantiles.loc[upper_threshold]
    asset_prices_pct[asset_prices_pct < quantiles.loc[lower_threshold]] = quantiles.loc[lower_threshold]

    return asset_prices_pct


def _compute_asset_performance(config: Config, asset_prices_pct: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the "asset market performance" using a rolling window sum.

    Calculates the asset market performance by applying a rolling window sum
    to the percentage changes in asset prices. The rolling window size and step are derived
    from the configiguration.

    Params:
        config:           Containing relevant settings for calculating the asset market performance.
        asset_prices_pct: A DataFrame containing the percentage change in asset prices.

    Returns:
        A DataFrame containing the computed asset market performance.
    """
    window = int(config.moving_window_increments)
    return asset_prices_pct.rolling(window=window, step=1, min_periods=window).sum(engine="numba")[window - 1 :]


"""
Default buy strategy

Recommended to pre-compute the asset_performance.
However, it is also possible to compute it on the fly,
this is much slower but more flexible to enable some use cases:

window_history_prices_pct = get_current_window(params, asset_prices_pct)
buyfactor_row = np.sum(window_history_prices_pct, axis=0)
"""


@nb.njit(cache=True, parallel=False)
def buy_strategy(params: dict, asset_prices_pct: np.ndarray, asset_performance: np.ndarray) -> np.ndarray:
    """
    Executes the buy strategy by selecting assets that meet specific conditions
    defined by the given parameters. The buy strategy is determined based on the
    asset performance score and user-defined preferences.

    Params:
        params:            Dict containing the parameters for the buy strategy, including:
                           - "row_idx":                    The current row index.
                           - "moving_window_increments":  The size of the rolling window.
                           - "buy_performance_score":      The target performance score for buying assets.
                           - "buy_performance_boundary":   The range around the target performance score.
                           - "buy_performance_preference": The user preference for buying assets.
        asset_prices_pct:  Array containing the percentage change in asset prices.
        asset_performance: Array containing the asset performance.

    Returns:
        Array containing the buy options for assets based on their performance
                and user-defined preferences.

    """
    row_idx = np.int64(params["row_idx"] - params["moving_window_increments"])
    buy_performance_scores = asset_performance[row_idx]
    buy_options_mask = _get_buy_options_mask(params, buy_performance_scores)

    return _get_buy_option_array(params, buy_performance_scores, buy_options_mask)


@nb.njit(cache=True, parallel=False)
def _get_buy_options_mask(params: dict, buy_performance_scores: np.ndarray) -> np.ndarray:
    """
    Generates a boolean mask representing assets that meet the specified buy performance score range.

    Calculate the min and max scores based on the given parameters and compare them
    with the buy_performance_scores. Returns a boolean numpy array which can be utilized as a mask.

    Params:
        params:        Dict containing the parameters for the buy strategy, including:
                       - "buy_performance_score": The target performance score for buying assets.
                       - "buy_performance_boundary": The range around the target performance score.
        buyfactor_row: Array containing the asset performance scores.

    Returns:
        A boolean numpy array (mask) where each element is True if the corresponding asset
        in the buy_performance_scores meets the buy performance score range, and False otherwise.
    """
    buy_performance_score_min = params["buy_performance_score"] - params["buy_performance_boundary"]
    buy_performance_score_max = params["buy_performance_score"] + params["buy_performance_boundary"]
    return (buy_performance_scores >= buy_performance_score_min) & (buy_performance_scores <= buy_performance_score_max)


@nb.njit(cache=True, parallel=False)
def _get_buy_option_array(params: dict, buy_performance_scores: np.ndarray, buy_options_mask: np.ndarray) -> np.ndarray:
    """
    Generates an array containing the buy options for assets
    based on their performance scores and user-defined preferences.

    The buy options are represented as indices of assets that meet the buy performance score range.
    The array is sorted based on the user's preference for buying assets.

    Params:
        params:                 Dict containing the parameters for the buy strategy, including:
                                - "buy_performance_score": The target performance score for buying assets.
                                - "buy_performance_boundary": The range around the target performance score.
                                - "buy_performance_preference": The user preference for buying assets.
        buy_performance_scores: Array containing all asset performance scores.
        buy_options_mask:       A boolean numpy array (mask) representing assets
                                that meet the specified buy performance score range.

    Returns:
        Array containing the buy options for assets as indices, sorted based on their performance scores and
        user-defined preferences.
    """
    buy_option_indices = np.argwhere(buy_options_mask).flatten().astype(np.float64)
    buy_option_values = buy_performance_scores[buy_options_mask]

    # Prefer values closer to the target performance score
    if params["buy_performance_preference"] == 0:
        buy_option_values = np.absolute(buy_option_values - params["buy_performance_score"])

    # Prefer values with lower performance scores (ascending order)
    buy_option_array = np.vstack((buy_option_indices, buy_option_values))
    buy_option_array = buy_option_array[:, buy_option_array[1, :].argsort()]

    # Prefer values with higher performance scores (descending order -> flip axis)
    if params["buy_performance_preference"] == 1:
        buy_option_array = buy_option_array[:, ::-1]

    return buy_option_array.astype(np.int64)[0]


"""
Default sell strategy
"""


@nb.njit(cache=True, parallel=False)
def sell_strategy(params: dict, buybag: np.ndarray, history_prices_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Executes the sell strategy by identifying assets
    that meet specific selling conditions based on the given parameters.

    The sell strategy is determined based on whether the assets have reached their target prices
    or if the hold time has elapsed and the current profit ratio meets the minimum profit.

    Params:
        params:             Dict containing the parameters for the sell strategy, including:
                            - "row_idx": The current row index.
                            - "_hold_time_increments": The minimum hold time before selling an asset.
                            - "profit_factor_target_min": The minimum profit factor required for selling an asset.
        buybag:             Array containing the buy bag, which stores buy times, buy prices, target prices etc.
        history_prices_row: Array containing the current prices of all assets in the market.

    Returns:
        A tuple containing two numpy arrays:
        - A boolean numpy array representing the sell decision for each asset in the buy bag.
            True indicates that the asset should be sold, and False indicates that the asset should be kept.
        - Array containing the current prices of the assets in the buy bag.
    """
    buy_option_indices = buybag[:, 0:1].T.flatten().astype(np.int64)
    current_prices = history_prices_row[buy_option_indices].reshape((1, -1)).T
    target_prices = buybag[:, 4:5]

    current_prices = _apply_sanity_check(current_prices, target_prices)

    times_since_buy = params["row_idx"] - buybag[:, 2:3]
    current_profit_ratios = current_prices / buybag[:, 3:4]

    sell_on_target_prices = current_prices >= target_prices
    sell_on_hold_time = _calculate_sell_on_hold_time(params, times_since_buy, current_profit_ratios)

    sell_assets = (sell_on_target_prices | sell_on_hold_time).flatten()

    return sell_assets, current_prices


@nb.njit(cache=True, parallel=False)
def _apply_sanity_check(current_prices: np.ndarray, target_prices: np.ndarray) -> np.ndarray:
    """
    Applies a sanity check on the current prices of assets by comparing them to a threshold.

    If the ratio of the current price to the target price is greater than a specified threshold (1.2 in this case),
    the current price is replaced with the target price. This helps prevent extreme or erroneous price values from
    affecting the sell strategy decisions.

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
def _calculate_sell_on_hold_time(
    params: dict, times_since_buy: np.ndarray, current_profit_ratios: np.ndarray
) -> np.ndarray:
    """
    Calculates the sell conditions based on hold time and the current profit ratios of the assets in the buy bag.

    Using the given parameter (profit_factor_target_min), this function determines if
    an asset should be sold based on the hold time (_hold_time_increments) and the current profit ratios.
    It checks if the time since the buy is greater the hold time (max_hold_time_reached) and
    if the current profit ratio is greater or equal to the minimum target profit factor (min_profit_factor_reached).

    Params:
        params:                Dict containing the parameters for the sell strategy, including:
                               - "_hold_time_increments": The number of increments for the hold time.
                               - "profit_factor_target_min": The minimum target profit factor.
        times_since_buy:       Array containing the time since each asset in the buy bag was bought.
        current_profit_ratios: Array containing the current profit ratios of the assets in the buy bag.

    Returns:
        A boolean numpy array where each element is True if the corresponding asset in the buy bag
        meets the hold time and profit ratio conditions for selling, and False otherwise."""
    max_hold_time_reached = times_since_buy > params["_hold_time_increments"]
    min_profit_factor_reached = current_profit_ratios >= params["profit_factor_target_min"]

    return max_hold_time_reached & min_profit_factor_reached


"""
Potential helper function for buy strategy
For reference see "Default buy strategy" docstring
"""


@nb.njit(cache=True, parallel=False)
def get_current_window(params: dict, history_prices_pct: pd.DataFrame) -> pd.DataFrame:
    moving_window_start = int(params["row_idx"] - params["moving_window_increments"])
    moving_window_end = int(params["row_idx"])
    return history_prices_pct.iloc[moving_window_start:moving_window_end]


def main() -> None:
    sim_result = Tradeforce(config=CONFIG).run_sim(
        pre_process=pre_process, buy_strategy=buy_strategy, sell_strategy=sell_strategy
    )
    print("Score (mean profit - std):", sim_result["score"])


if __name__ == "__main__":
    main()
