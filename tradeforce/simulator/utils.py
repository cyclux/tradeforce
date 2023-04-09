"""_summary_
"""

import numpy as np
import numba as nb  # type: ignore
import numba.typed as nb_types
from numba.core import types  # type: ignore

NB_PARALLEL = False
NB_CACHE = True


# legacy params, but may be useful with other params
def numba_dict_defaults(sim_params):
    set_default_params = [
        "buy_performance_score",
        "buy_performance_boundary",
        "buy_performance_score_min",
        "buy_performance_score_max",
    ]
    for default_param in set_default_params:
        sim_params[default_param] = sim_params.get(default_param, 999)
    return sim_params


def to_numba_dict(sim_params):
    # sim_params = numba_dict_defaults(sim_params)
    sim_params_numba = nb_types.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    for key, val in sim_params.items():
        sim_params_numba[key] = np.float64(val)
    return sim_params_numba


# TODO: Do this before numba compilation and pre_process
@nb.njit(cache=NB_CACHE, parallel=False)
def check_min__subset_size_increments(_subset_size_increments, subset_amount, subset_idx_boundary):
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
def sanitize_subset_params(params, subset_idx_boundary):
    _subset_size_increments = params["_subset_size_increments"]
    subset_amount = np.int64(params["subset_amount"])
    if _subset_size_increments <= 0:
        _subset_size_increments = -1
    if subset_amount <= 0:
        subset_amount = 1
    if subset_amount == 1 and _subset_size_increments == -1:
        _subset_size_increments = subset_idx_boundary
    if subset_amount > 1 and _subset_size_increments == -1:
        _subset_size_increments = subset_idx_boundary // subset_amount
    check_min__subset_size_increments(_subset_size_increments, subset_amount, subset_idx_boundary)
    return _subset_size_increments, subset_amount


# currently not used
@nb.njit(cache=NB_CACHE, parallel=False)
def array_diff(arr1, arr2):
    diff_list = nb_types.List(set(arr1) - set(arr2))
    diff_array = np.array([x for x in diff_list])
    return diff_array


# currently not used
@nb.njit(cache=NB_CACHE, parallel=False)  # parallel not checked
def fill_nan(nd_array):
    shape = nd_array.shape
    nd_array = nd_array.ravel()
    nd_array[np.isnan(nd_array)] = 0
    nd_array = nd_array.reshape(shape)
    return nd_array


@nb.njit(cache=NB_CACHE, parallel=False)
def calc_fee(volume_asset, maker_fee, taker_fee, prices_current, order_type):
    volume_asset = np.absolute(volume_asset)
    exchange_fee = taker_fee if order_type == "buy" else maker_fee
    amount_fee_asset = (volume_asset / 100) * exchange_fee
    volume_asset_incl_fee = volume_asset - amount_fee_asset
    amount_fee_fiat = amount_fee_asset * prices_current
    return volume_asset_incl_fee, amount_fee_asset, amount_fee_fiat


@nb.njit(cache=NB_CACHE, parallel=False)
def get_subset_indices(
    _moving_window_increments,
    subset_idx_boundary,
    subset_amount=10,
    _subset_size_increments=10000,
):
    subset_idx_boundary = np.int64(subset_idx_boundary - _subset_size_increments)
    subset_idxs = np.linspace(_moving_window_increments, subset_idx_boundary, subset_amount).astype(np.int64)
    return subset_idxs


@nb.njit(cache=NB_CACHE, parallel=NB_PARALLEL)
def calc_metrics(soldbag):
    total_profit = soldbag[:, 14:15].sum()
    return np.int64(total_profit)


@nb.njit(cache=NB_CACHE, parallel=True)
def get_pct_change(df_history_prices):
    df_history_prices_pct = (df_history_prices[1:, :] - df_history_prices[:-1, :]) / df_history_prices[1:, :]
    amount_zeros = df_history_prices_pct.shape[-1]
    zeros = np.zeros((1, amount_zeros))
    df_history_prices_pct = np.vstack((zeros, df_history_prices_pct))
    return df_history_prices_pct


@nb.njit(cache=NB_CACHE, parallel=False)
def get_current_window(params, df_history_prices_pct):
    moving_window_start = np.int64(params["row_idx"] - params["_moving_window_increments"])
    moving_window_end = np.int64(params["row_idx"])
    moving_window_history_prices_pct = df_history_prices_pct[moving_window_start:moving_window_end]
    return moving_window_history_prices_pct
