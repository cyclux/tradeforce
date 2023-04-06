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

Returns:
    _type_: _description_
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import numba as nb  # type: ignore
import pandas as pd
import tradeforce.simulator.utils as sim_utils
from tradeforce.utils import get_timedelta
from tradeforce.simulator.buys import check_buy
from tradeforce.simulator.sells import check_sell
from tradeforce.simulator.default_strategies import pre_process

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce import Tradeforce

type_float = nb.typeof(0.1)


NB_PARALLEL = False
NB_CACHE = True


@nb.njit(cache=NB_CACHE, parallel=False)
def set_budget_from_bag(row_idx, budget, bag, bag_type):
    # bag_row_idx of 2 == row_idx of buy opportunity
    # bag_row_idx of 9 == row_idx of sell opportunity
    bag_row_idx = 2 if bag_type == "buybag" else 9
    if bag.shape[0] > 0:
        row_idx_last_tx = bag[-1, bag_row_idx]
        if row_idx_last_tx == row_idx:
            budget = bag[-1, 8]
    return budget


# cache needs to be off in case of a user defined trading strategy function
# @perf.jit_timer
@nb.njit(cache=False, parallel=False)
def iter_market_history(
    params, snapshot_bounds, asset_prices, asset_prices_pct, asset_performance
) -> tuple[np.ndarray, np.ndarray]:
    budget = params["budget"]
    buybag = np.empty((0, 9), type_float)
    soldbag = np.empty((0, 15), type_float)
    start_idx = snapshot_bounds[0]
    end_idx = snapshot_bounds[1]
    for row_idx, history_prices_row in enumerate(asset_prices[start_idx:end_idx]):
        # row_idx needs to get shifted by start_idx (start idx of the current snapshot iteration)
        row_idx += start_idx
        params["row_idx"] = row_idx

        soldbag, buybag = check_sell(params, buybag, soldbag, history_prices_row, budget)
        budget = set_budget_from_bag(row_idx, budget, soldbag, "soldbag")
        buybag = check_buy(params, asset_prices_pct, asset_performance, buybag, history_prices_row, budget)
        budget = set_budget_from_bag(row_idx, budget, buybag, "buybag")

    return soldbag, buybag


# cache needs to be off in case of a user defined trading strategy function
@nb.njit(cache=False, parallel=False)
def simulate_trading(
    params, asset_prices, asset_prices_pct, asset_performance
) -> tuple[np.int64, np.ndarray, np.ndarray]:

    snapshot_idx_boundary = asset_prices.shape[0]
    snapshot_size, snapshot_amount = sim_utils.sanitize_snapshot_params(params, snapshot_idx_boundary)
    snapshot_start_idxs = sim_utils.get_snapshot_indices(
        params["moving_window_increments"], snapshot_idx_boundary, snapshot_amount, snapshot_size
    )
    profit_snapshot_list = np.empty(snapshot_amount, type_float)
    soldbag_all_snapshots = np.empty((0, 15), type_float)
    # initialize row_idx within params -> provide access to sub-functions
    params["row_idx"] = 0.0
    for current_iter, snapshot_idx in enumerate(snapshot_start_idxs):
        snapshot_bounds = (snapshot_idx, snapshot_idx + snapshot_size)

        soldbag, buybag = iter_market_history(
            params, snapshot_bounds, asset_prices, asset_prices_pct, asset_performance
        )

        profit_snapshot_list[current_iter] = sim_utils.calc_metrics(soldbag)
        soldbag_all_snapshots = np.vstack((soldbag_all_snapshots, soldbag))

    profit_total_std = np.std(profit_snapshot_list)
    profit_total = np.int64(np.mean(profit_snapshot_list) - profit_total_std)

    print("profit_total:", profit_total)

    return profit_total, soldbag_all_snapshots, buybag


def print_sim_details(root: Tradeforce, asset_prices: pd.DataFrame):
    history_begin = asset_prices.index[0]
    history_end = asset_prices.index[-1]
    history_delta = get_timedelta(history_end - history_begin, unit="ms")["datetime"]  # type: ignore
    root.log.info(
        "Starting simulation beginning from %s to %s | Timeframe: %s", history_begin, history_end, history_delta
    )
    root.log.info(
        "Simulation is split into %s snapshots with size %s each.",
        root.config.snapshot_amount,
        root.config.snapshot_size,
    )


def run(root: Tradeforce) -> dict[str, int | np.ndarray | np.ndarray]:
    preprocess_result = pre_process(root.config, root.market_history)
    sim_config = sim_utils.to_numba_dict(root.config.to_dict())

    asset_prices = preprocess_result.get("asset_prices", pd.DataFrame())
    asset_prices_pct = preprocess_result.get("asset_prices_pct", pd.DataFrame())
    asset_performance = preprocess_result.get("asset_performance", pd.DataFrame())

    print_sim_details(root, asset_prices)

    if asset_prices is not None:
        np_asset_prices = asset_prices.to_numpy()
    if asset_prices_pct is not None:
        np_asset_prices_pct = asset_prices_pct.to_numpy()
    if asset_performance is not None:
        np_asset_performance = asset_performance.to_numpy()

    total_profit, trades_history, buy_log = simulate_trading(
        sim_config, np_asset_prices, np_asset_prices_pct, np_asset_performance
    )
    sim_result = {"profit": total_profit, "trades": trades_history, "buy_log": buy_log}
    return sim_result
