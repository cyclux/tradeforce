""" simulator/utils.py

Module: tradeforce.simulator.utils
----------------------------------

Provides utility functions for the TradeForce simulator, including converting dictionaries
to Numba typed dictionaries, checking and sanitizing parameters, calculating fees and metrics,
and generating subset slice indices. It also includes functions for array manipulation, such as
calculating the percent change between consecutive rows in a NumPy array and replacing NaN
values with zeros.

Functions:

    to_numba_dict: Convert a Python dictionary to a Numba typed dictionary.
    _check_min_subset_size_increments: Check if a subset size increment value is within acceptable limits.
    sanitize_subset_params: Sanitize subset parameters to ensure they are within acceptable limits.
    array_diff: Calculate the difference between two arrays.
    calc_fee: Calculate the fee for a given order.
    get_subset_indices: Generate an array of subset indices.
    calc_metrics: Calculate the total profit metric from the sold bag.
    get_pct_change: Calculate the percent change between consecutive rows in a NumPy array.
    fill_nan: Replace NaN values in a NumPy array with zeros.

"""

import numpy as np
import numba as nb  # type: ignore
import numba.typed as nb_types
from numba.core import types  # type: ignore

NB_PARALLEL = False
NB_CACHE = True


def to_numba_dict(sim_params: dict) -> nb_types.Dict[types.unicode_type, types.float64]:
    """Convert a Python dictionary to a Numba typed dictionary.

    Params:
        sim_params: A dictionary with keys as strings and values as float64.

    Returns:
        A Numba typed dictionary containing the same key-value pairs as the input dictionary.
    """
    sim_params_numba = nb_types.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    for key, val in sim_params.items():
        sim_params_numba[key] = np.float64(val)

    return sim_params_numba


# TODO: Do this before numba compilation and pre_process
@nb.njit(cache=NB_CACHE, parallel=False)
def _check_min_subset_size_increments(
    _subset_size_increments: np.float64, subset_amount: np.float64, subset_idx_boundary: np.float64
) -> np.float64:
    """Check if the _subset_size_increments value is within acceptable limits.

    This function prints warnings or errors if the _subset_size_increments value is too small,
    less than 1, or greater than subset_idx_boundary.

    Params:
        _subset_size_increments: The size increment of each subset.
        subset_amount: The total number of subsets.
        subset_idx_boundary: The upper boundary for subset indices.

    Returns:
        The _subset_size_increments value if it is within acceptable limits.
    """
    if _subset_size_increments < 100:
        print("[WARNING]: _subset_size_increments is very small: ", _subset_size_increments)
    if _subset_size_increments < 1:
        print(
            "[ERROR] _subset_size_increments is",
            int(_subset_size_increments),
            ".Please check your parameters.",
            "subset_amount:",
            int(subset_amount),
            "| subset_idx_boundary: ",
            int(subset_idx_boundary),
        )
    if _subset_size_increments > subset_idx_boundary:
        print(
            "[ERROR] _subset_size_increments > subset_idx_boundary:",
            int(_subset_size_increments),
            ">",
            int(subset_idx_boundary),
            "| subset_amount:",
            int(subset_amount),
        )
    return _subset_size_increments


@nb.njit(cache=NB_CACHE, parallel=False)
def sanitize_subset_params(params: dict, subset_idx_boundary: np.float64) -> tuple:
    """Sanitize the subset parameters to ensure they are within acceptable limits.

    Params:
        params: Dict containing _subset_size_increments and subset_amount keys.
        subset_idx_boundary: The upper boundary for subset indices.

    Returns:
        A tuple containing the sanitized values of _subset_size_increments and subset_amount.
    """
    _subset_size_increments = np.int64(params["_subset_size_increments"])
    subset_amount = np.int64(params["subset_amount"])
    if _subset_size_increments <= 0:
        _subset_size_increments = np.int64(-1)
    if subset_amount <= 0:
        subset_amount = np.int64(1)
    if subset_amount == 1 and _subset_size_increments == -1:
        _subset_size_increments = np.int64(subset_idx_boundary)
    if subset_amount > 1 and _subset_size_increments == -1:
        _subset_size_increments = np.int64(subset_idx_boundary) // subset_amount
    _check_min_subset_size_increments(_subset_size_increments, subset_amount, subset_idx_boundary)
    return _subset_size_increments, subset_amount


# currently not used
@nb.njit(cache=NB_CACHE, parallel=False)
def array_diff(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Calculate the difference between two arrays.

    This function finds the set difference between two arrays
    and returns the result as a new array.

    Params:
        arr1: The first array.
        arr2: The second array.

    Returns:
        A new array containing the elements that are unique to arr1.
    """
    diff_list = nb_types.List(set(arr1) - set(arr2))
    diff_array = np.array([x for x in diff_list])
    return diff_array


@nb.njit(cache=NB_CACHE, parallel=False)
def calc_fee(
    volume_asset: np.int64, maker_fee: np.float64, taker_fee: np.float64, prices_current: np.ndarray, order_type: str
) -> tuple:
    """Calculate the fee for a given order.

    Params:
        volume_asset: The volume of the asset.
        maker_fee: The maker fee percentage.
        taker_fee: The taker fee percentage.
        prices_current: The current price of the asset.
        order_type: A string representing the order type ("buy" or "sell").

    Returns:
        A tuple containing the volume of the asset including the fee,
        the amount of the fee in the asset, and the amount of the fee in fiat currency.
    """
    volume_asset = np.absolute(volume_asset)
    exchange_fee = taker_fee if order_type == "buy" else maker_fee
    amount_fee_asset = (volume_asset / 100) * exchange_fee
    volume_asset_incl_fee = volume_asset - amount_fee_asset
    amount_fee_fiat = amount_fee_asset * prices_current
    return volume_asset_incl_fee, amount_fee_asset, amount_fee_fiat


@nb.njit(cache=NB_CACHE, parallel=False)
def get_subset_indices(
    moving_window_increments: np.float64,
    subset_idx_boundary: np.float64,
    subset_amount: np.int64,
    _subset_size_increments: np.int64,
) -> np.ndarray:
    """Generate an array of subset indices.

    Params:
        moving_window_increments: The moving window increments.
        subset_idx_boundary: The upper boundary for subset indices.
        subset_amount: The total number of subsets.
        _subset_size_increments: The size increment of each subset.

    Returns:
        A NumPy array containing the subset indices.
    """
    subset_idx_boundary_int = np.int64(subset_idx_boundary - _subset_size_increments)
    subset_idxs = np.linspace(moving_window_increments, subset_idx_boundary_int, subset_amount).astype(np.int64)
    return subset_idxs


@nb.njit(cache=NB_CACHE, parallel=NB_PARALLEL)
def calc_metrics(soldbag: np.ndarray) -> np.int64:
    """Calculate the total profit metric from the sold bag.

    Params:
        soldbag: A NumPy array containing information about sold assets.
                The 15th column (index 14) should contain the profit values.

    Returns:
        The total profit as an int64 value.
    """
    total_profit = soldbag[:, 14:15].sum()
    return np.int64(total_profit)


@nb.njit(cache=NB_CACHE, parallel=True)
def get_pct_change(df_history_prices: np.ndarray) -> np.ndarray:
    """Calculate the percent change between consecutive rows in a NumPy array.

    Params:
        df_history_prices: A NumPy array containing historical prices.

    Returns:
        A NumPy array with the same shape as the input array, containing the percent change
        between consecutive rows. The first row of the output array will be filled with zeros.
    """
    df_history_prices_pct = (df_history_prices[1:, :] - df_history_prices[:-1, :]) / df_history_prices[1:, :]
    amount_zeros = df_history_prices_pct.shape[-1]
    zeros = np.zeros((1, amount_zeros))
    df_history_prices_pct = np.vstack((zeros, df_history_prices_pct))
    return df_history_prices_pct


# currently not used
@nb.njit(cache=NB_CACHE, parallel=False)
def fill_nan(nd_array: np.ndarray) -> np.ndarray:
    """Replace NaN values in a NumPy array with zeros.

    Params:
        nd_array: A NumPy array potentially containing NaN values.

    Returns:
        A NumPy array with NaN values replaced by zeros.
    """
    shape = nd_array.shape
    nd_array = nd_array.ravel()
    nd_array[np.isnan(nd_array)] = 0
    nd_array = nd_array.reshape(shape)
    return nd_array
