"""_summary_
# 0 idx of asset row -> symbol of asset
# 1 buy opportunity factor
# 2 row_idx of buy opportunity
# 3 price buy
# 4 price including profit
# 5 amount of fiat invested, fee included
# 6 amount of crypto invested, fee included
# 7 amount fee buy order in fiat (probably taker fee)
# 8 budget to invest
#
# 9 row_idx of sell opportunity
# 10 price sell
# 11 amount of fiat sold, fee included
# 12 amount of crypto sold, fee included
# 13 amount fee sell order in fiat (probably maker fee)
# 14 amount profit in fiat
# 15 value_crypto_in_fiat
# 16 total value of account
# 17 amount buy orders
# 18 current snapshot iteration
# 19 current_idx

Returns:
    _type_: _description_
"""

import numpy as np
import numba as nb
import numba.typed as nb_types
from fts_utils import to_numba_dict


# FIXME: Add money which is in assets on the market

# type_int = nb.typeof(1)
type_float = nb.typeof(0.1)
# type_array_1d_float = nb.typeof(np.array([0.1]))
# type_array_2d_float = nb.typeof(np.array([[0.1]]))

NB_PARALLEL = False
NB_CACHE = True


@nb.njit(cache=NB_CACHE, parallel=False)
def array_diff(arr1, arr2):
    diff_list = nb_types.List(set(arr1) - set(arr2))
    diff_array = np.array([x for x in diff_list])
    return diff_array


@nb.njit(cache=NB_CACHE, parallel=False)  # parallel not checked
def fill_nan(nd_array):
    shape = nd_array.shape
    nd_array = nd_array.ravel()
    nd_array[np.isnan(nd_array)] = 0
    nd_array = nd_array.reshape(shape)
    return nd_array


@nb.njit()
def get_buy_option_assets(buy_options_bool, buyfactor_row, buy_opportunity_factor, prefer_performance):
    # prefer_performance can be -1, 1, and 0.
    buy_option_indices = np.where(buy_options_bool)[0].astype(np.float64)
    buy_option_values = buyfactor_row[buy_options_bool]
    # buy_opportunity_factor == 999 means it is not set
    if prefer_performance == 0 and buy_opportunity_factor != 999:
        buy_option_values = np.absolute(buy_option_values - buy_opportunity_factor)
    buy_option_array = np.vstack((buy_option_indices, buy_option_values))
    buy_option_array = buy_option_array[:, buy_option_array[1, :].argsort()]
    if prefer_performance == 1:
        # flip axis
        buy_option_array = buy_option_array[:, ::-1]
    buy_option_array_int = buy_option_array[0].astype(np.int64)
    return buy_option_array_int


@nb.njit(cache=NB_CACHE, parallel=False)
def calc_fee(volume, exchange_fee, price_current, currency_type="crypto"):
    fee_to_pay = volume / 100 * exchange_fee
    volume_incl_fee = volume - fee_to_pay
    if currency_type == "crypto":
        amount_fee_fiat = np.round(fee_to_pay * price_current, 2)
    if currency_type == "fiat":
        amount_fee_fiat = fee_to_pay
    return volume_incl_fee, amount_fee_fiat


@nb.njit(cache=NB_CACHE, parallel=False)
def get_snapshot_indices(snapshot_idx_boundary, snapshot_amount=10, snapshot_size=10000):
    snapshot_idx_boundary = np.int64(snapshot_idx_boundary - snapshot_size)
    snapshot_idxs = np.linspace(0, snapshot_idx_boundary, snapshot_amount).astype(np.int64)
    return snapshot_idxs


@nb.njit(cache=NB_CACHE, parallel=NB_PARALLEL)
def buy_asset(
    buy_option_idx, buyfactor_row, row_idx, price_current, profit_factor, amount_invest_fiat, exchange_fee, budget
):
    price_profit = price_current * profit_factor
    amount_invest_fiat_incl_fee, amount_fee_buy_fiat = calc_fee(
        amount_invest_fiat, exchange_fee, price_current, currency_type="fiat"
    )
    amount_invest_crypto_incl_fee = amount_invest_fiat_incl_fee / price_current
    budget -= amount_invest_fiat
    bought_asset_params = np.array(
        [
            [
                buy_option_idx,
                buyfactor_row[buy_option_idx],
                row_idx,
                price_current,
                price_profit,
                amount_invest_fiat_incl_fee,
                amount_invest_crypto_incl_fee,
                amount_fee_buy_fiat,
                budget,
            ]
        ]
    )
    return bought_asset_params, budget


@nb.njit(cache=NB_CACHE, parallel=NB_PARALLEL)
def check_buy(
    list_buy_options,
    buybag,
    asset_buy_limit,
    buy_limit_strategy,
    max_buy_per_asset,
    buyfactor_row,
    row_idx,
    df_history_prices,
    buy_opportunity_factor_max,
    profit_factor,
    amount_invest_fiat,
    exchange_fee,
    budget,
):
    for buy_option_idx in list_buy_options:
        price_current = df_history_prices[row_idx, buy_option_idx]
        if buy_option_idx in buybag[:, 0:1]:

            buybag_idxs = buybag[:, 0:1]
            buybag_idx = np.where(buybag_idxs == buy_option_idx)[0]
            asset_buy_prices = buybag[buybag_idx, 3]
            min_price_bought = np.min(asset_buy_prices)

            is_max_buy_per_asset = len(asset_buy_prices) >= max_buy_per_asset

            is_buy_opportunity = min_price_bought * (1 + buy_opportunity_factor_max) >= price_current
            if is_max_buy_per_asset or not is_buy_opportunity:
                continue

        amount_buy_orders = buybag.shape[0]
        buy_condition = amount_buy_orders < asset_buy_limit if buy_limit_strategy == 1 else budget >= amount_invest_fiat
        if buy_condition:
            bought_asset_params, budget = buy_asset(
                buy_option_idx,
                buyfactor_row,
                row_idx,
                price_current,
                profit_factor,
                amount_invest_fiat,
                exchange_fee,
                budget,
            )
            buybag = np.append(buybag, bought_asset_params, axis=0)
    return buybag


@nb.njit(cache=NB_CACHE, parallel=NB_PARALLEL)
def sell_asset(
    current_iter,
    current_idx,
    row_idx,
    bought_asset_params,
    soldbag,
    price_current,
    amount_invest_fiat,
    exchange_fee,
    budget,
):

    amount_invest_crypto = bought_asset_params[6]
    amount_sold_crypto_incl_fee, amount_fee_sell_fiat = calc_fee(
        amount_invest_crypto, exchange_fee, price_current, currency_type="crypto"
    )
    amount_sold_fiat_incl_fee = np.round(amount_sold_crypto_incl_fee * price_current, 2)
    amount_profit_fiat = amount_sold_fiat_incl_fee - amount_invest_fiat
    bought_asset_params[8] = budget + amount_sold_fiat_incl_fee
    placeholder_value_crypto_in_fiat = 0.0
    placeholder_total_value = 0.0
    placeholder_amount_buy_orders = 0.0
    sold_asset_params = np.array(
        [
            row_idx,
            price_current,
            amount_sold_fiat_incl_fee,
            amount_sold_crypto_incl_fee,
            amount_fee_sell_fiat,
            amount_profit_fiat,
            placeholder_value_crypto_in_fiat,
            placeholder_total_value,
            placeholder_amount_buy_orders,
            current_iter,
            current_idx,
        ]
    )
    bought_asset_params = np.append(bought_asset_params, sold_asset_params)
    bought_asset_params = np.expand_dims(bought_asset_params, axis=0)

    # print(
    #     "profit",
    #     amount_profit_fiat,
    #     "sell_vol_fiat",
    #     amount_sold_fiat_incl_fee,
    #     "sell_vol_crypto",
    #     amount_sold_crypto_incl_fee,
    #     "price_sell",
    #     price_current,
    # )

    soldbag = np.append(soldbag, bought_asset_params, axis=0)
    return soldbag


@nb.njit(cache=NB_CACHE, parallel=NB_PARALLEL)
def check_sell(
    current_iter,
    current_idx,
    buybag,
    soldbag,
    row_idx,
    df_history_prices,
    amount_invest_fiat,
    exchange_fee,
    budget,
    asset_buy_limit,
    buy_limit_strategy,
    hold_time_limit,
    profit_ratio_limit,
):
    if buybag.shape[0] < 1:
        return soldbag, buybag

    del_buybag_items_list = nb_types.List()
    amount_buy_orders = buybag.shape[0]
    for buybag_row_idx, bought_asset_params in enumerate(buybag):
        buy_option_idx = bought_asset_params[0]
        buy_option_idx_int = np.int64(buy_option_idx)
        row_idx_bought = bought_asset_params[2]
        price_current = df_history_prices[row_idx, buy_option_idx_int]
        price_bought = bought_asset_params[3]
        price_profit = bought_asset_params[4]
        time_since_buy = row_idx - row_idx_bought
        current_profit_ratio = price_current / price_bought

        buy_orders_maxed_out = (
            amount_buy_orders >= asset_buy_limit if buy_limit_strategy == 1 else budget < amount_invest_fiat
        )
        ok_to_sell = (
            time_since_buy > hold_time_limit and current_profit_ratio >= profit_ratio_limit and buy_orders_maxed_out
        )
        if (price_current >= price_profit) or ok_to_sell:
            # check plausibility and prevent false logic
            # profit gets a max plausible threshold
            if price_current / price_profit > 1.2:
                price_current = price_profit
            soldbag = sell_asset(
                current_iter,
                current_idx,
                row_idx,
                bought_asset_params,
                soldbag,
                price_current,
                amount_invest_fiat,
                exchange_fee,
                budget,
            )
            budget = soldbag[-1, 8]
            del_buybag_items_list.append(buybag_row_idx)

    if len(del_buybag_items_list) > 0:
        del_buybag_items_array = np.array([x for x in del_buybag_items_list])
        item_row_idx_range = np.array([x for x in range(amount_buy_orders)])
        excluded_del_rows = array_diff(item_row_idx_range, del_buybag_items_array)
        buybag = buybag[excluded_del_rows, :]

    for sold_asset_idx in range(len(del_buybag_items_list)):
        # Calculate buybag crypto value in fiat
        crypto_invested = buybag[:, 6:7].flatten()
        asset_idx = buybag[:, 0:1].flatten().astype(np.int64)
        # TODO: Linter quick fix: whitespace : row_idx + 1
        row_idx_1 = row_idx + 1
        current_price_buybag = df_history_prices[row_idx:row_idx_1, asset_idx].flatten()
        value_crypto_in_fiat = np.sum(current_price_buybag * crypto_invested)
        sold_asset_reverse_idx = (sold_asset_idx * -1) - 1
        soldbag[sold_asset_reverse_idx:, 15:16] = np.round(value_crypto_in_fiat, 2)
        soldbag[sold_asset_reverse_idx:, 16:17] = np.round(value_crypto_in_fiat + budget, 2)
        soldbag[sold_asset_reverse_idx:, 17:18] = buybag.shape[0]

    return soldbag, buybag


@nb.njit(cache=NB_CACHE, parallel=NB_PARALLEL)
def calc_metrics(soldbag):
    total_profit = soldbag[:, 14:15].sum()
    return np.int64(total_profit)


@nb.njit(cache=NB_CACHE, parallel=False)
def iter_market_history(
    df_buy_factors,
    snapshot_bounds,
    df_history_prices,
    amount_invest_fiat,
    budget,
    buy_opportunity_factor,
    buy_opportunity_factor_max,
    buy_opportunity_factor_min,
    prefer_performance,
    profit_factor,
    asset_buy_limit,
    buy_limit_strategy,
    max_buy_per_asset,
    hold_time_limit,
    profit_ratio_limit,
    exchange_fee,
    current_idx,
    current_iter,
):
    buybag = np.empty((0, 9), type_float)
    soldbag = np.empty((0, 20), type_float)
    start_idx = snapshot_bounds[0]
    end_idx = snapshot_bounds[1]
    for row_idx, buyfactor_row in enumerate(df_buy_factors[start_idx:end_idx]):
        soldbag, buybag = check_sell(
            current_iter,
            current_idx,
            buybag,
            soldbag,
            row_idx,
            df_history_prices,
            amount_invest_fiat,
            exchange_fee,
            budget,
            asset_buy_limit,
            buy_limit_strategy,
            hold_time_limit,
            profit_ratio_limit,
        )

        if soldbag.shape[0] > 0:
            row_idx_last_sell = soldbag[-1, 9]
            if row_idx_last_sell == row_idx:
                budget = soldbag[-1, 8]

        buy_options_bool = (buyfactor_row >= buy_opportunity_factor_min) & (buyfactor_row <= buy_opportunity_factor_max)
        if np.any(buy_options_bool):
            list_buy_options = get_buy_option_assets(
                buy_options_bool, buyfactor_row, buy_opportunity_factor, prefer_performance
            )
            buybag = check_buy(
                list_buy_options,
                buybag,
                asset_buy_limit,
                buy_limit_strategy,
                max_buy_per_asset,
                buyfactor_row,
                row_idx,
                df_history_prices,
                buy_opportunity_factor_max,
                profit_factor,
                amount_invest_fiat,
                exchange_fee,
                budget,
            )
            if buybag.shape[0] > 0:
                row_idx_last_buy = buybag[-1, 2]
                if row_idx_last_buy == row_idx:
                    budget = buybag[-1, 8]

        current_idx += 300000

    return soldbag, buybag


@nb.njit(cache=NB_CACHE, parallel=False)
def simulate_trading(sim_params_numba, df_buy_factors, df_history_prices):
    current_idx = sim_params_numba["index_start"]
    window = sim_params_numba["window"] * 60 // 5
    buy_opportunity_factor_max = sim_params_numba["buy_opportunity_factor_max"]
    buy_opportunity_factor_min = sim_params_numba["buy_opportunity_factor_min"]
    buy_opportunity_factor = sim_params_numba["buy_opportunity_factor"]
    buy_opportunity_boundary = sim_params_numba["buy_opportunity_boundary"]
    prefer_performance = sim_params_numba["prefer_performance"]
    profit_factor = sim_params_numba["profit_factor"]
    budget = sim_params_numba["budget"]
    amount_invest_fiat = sim_params_numba["amount_invest_fiat"]
    exchange_fee = sim_params_numba["exchange_fee"]  # in percent -> maker 0.1 | taker 0.2
    buy_limit_strategy = sim_params_numba["buy_limit"]
    hold_time_limit = sim_params_numba["hold_time_limit"]
    profit_ratio_limit = sim_params_numba["profit_ratio_limit"]
    max_buy_per_asset = sim_params_numba["max_buy_per_asset"]
    snapshot_size = sim_params_numba["snapshot_size"]
    snapshot_amount = np.int64(sim_params_numba["snapshot_amount"])

    asset_buy_limit = budget // amount_invest_fiat
    if buy_opportunity_factor != 999:
        buy_opportunity_factor_min = buy_opportunity_factor - buy_opportunity_boundary
        buy_opportunity_factor_max = buy_opportunity_factor + buy_opportunity_boundary

    # Fill NaN probably not needed as it is done by the data fetch api
    # fill_nan(df_buy_factors)

    df_buy_factors = df_buy_factors[window:]
    df_history_prices = df_history_prices[window:]
    # current_idx += window * 300000

    snapshot_idx_boundary = df_buy_factors.shape[0]
    if snapshot_amount == 1 and snapshot_size == -1:
        snapshot_size = snapshot_idx_boundary
    snapshot_bounds = get_snapshot_indices(snapshot_idx_boundary, snapshot_amount, snapshot_size)

    profit_snapshot_list = np.empty(snapshot_amount, type_float)
    soldbag_all_snapshots = np.empty((0, 20), type_float)

    # index_start = np.float64(history_buy_factors.index[0])
    for current_iter, snapshot_idx in enumerate(snapshot_bounds, 1):
        snapshot_bounds = (snapshot_idx, snapshot_idx + snapshot_size)
        # TODO: DEBUG
        # print(snapshot_bounds)
        soldbag, buybag = iter_market_history(
            df_buy_factors,
            snapshot_bounds,
            df_history_prices,
            amount_invest_fiat,
            budget,
            buy_opportunity_factor,
            buy_opportunity_factor_max,
            buy_opportunity_factor_min,
            prefer_performance,
            profit_factor,
            asset_buy_limit,
            buy_limit_strategy,
            max_buy_per_asset,
            hold_time_limit,
            profit_ratio_limit,
            exchange_fee,
            current_idx + (snapshot_idx * 300000),
            current_iter,
        )
        profit_snapshot_list[current_iter - 1] = calc_metrics(soldbag)
        soldbag_all_snapshots = np.vstack((soldbag_all_snapshots, soldbag))

    profit_total_std = np.std(profit_snapshot_list)
    profit_total = np.int64(np.mean(profit_snapshot_list) - profit_total_std)

    return profit_total, soldbag_all_snapshots, buybag


def run_simulation(sim_params, market_history, market_history_pct):
    window = int(sim_params["window"] * 60 // 5)
    history_buy_factors = market_history_pct.rolling(window=window, step=1, min_periods=1).sum(
        engine="numba", engine_kwargs={"parallel": True, "cache": True}
    )
    total_profit, trades_history, buy_log = simulate_trading(
        to_numba_dict(sim_params), history_buy_factors.to_numpy(), market_history.to_numpy()
    )
    return total_profit, trades_history, buy_log
