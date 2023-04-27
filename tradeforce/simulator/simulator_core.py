""" simulator/simulator_core.py

Module: tradeforce.simulator
----------------------------

Provides the core functionality for running trading simulations using Tradeforce.
It defines the main entry point, run(), which takes a Tradeforce instance as
input, preprocesses the market history data, and performs trading simulations using
Numba JIT-compiled functions. The module includes functions to iterate through market
history within specified subsets, perform buy and sell operations, update budgets based
on transactions, and calculate the score based on the mean profit corrected by standard
deviation of profits across all subsets. The module also provides utility functions to
sanitize subset parameters, calculate and print simulation details, and convert data
structures to numpy arrays for use with Numba.

The np.array "buybag" contains the details of all buy transactions and is used to
store asset attributes, such as amount and profit ratio. Those attributes also help
to determine the elapsed time since buy or keep track of the available budget.

The np.array "soldbag" contains the details of all sell transactions, which also include
all the details from the buybag. The "soldbag" is used to calculate the actual profit
of each of the sold assets and stores sell relevant attributes, like sell price and time.


Index reference for the buybag and soldbag arrays:
    # Only buybag:
    [0] idx of asset row -> symbol of asset
    [1] buy_signal_score
    [2] row_idx at buy time
    [3] price buy
    [4] price including profit
    [5] amount of fiat invested, fee included
    [6] amount of asset invested, fee included
    [7] amount fee buy order in fiat (probably taker fee)
    [8] budget to invest

    # soldbag also includes:
    [9] row_idx of sell opportunity
    [10]  price sell
    [11]  amount of fiat sold, fee included
    [12]  amount of asset sold, fee included
    [13]  amount fee sell order in fiat (probably maker fee)
    [14]  amount profit in fiat

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

# Determine the type of numba float
type_float = nb.typeof(0.1)


NB_PARALLEL = False
NB_CACHE = True


@nb.njit(cache=NB_CACHE, parallel=False)
def update_budget_from_last_transaction(row_idx: int, budget: float, bag: np.ndarray, bag_type: str) -> float:
    """Update the budget

    based on the last transaction in the bag if the row_idx of the last transaction matches the
    current row_idx. Ensures that the budget accurately reflects the latest transaction within the
    same row index in the given bag (buy or sell).

    Params:
        row_idx:  The current row index representing the position in the market history.
                    This is used to determine whether the budget should be updated based
                    on the last transaction.

        budget:   The current budget, which represents the amount of funds available for trading.

        bag:      The transaction bag (buy or sell), a numpy array containing
                    the details of each transaction.

        bag_type: The type of bag, either "buybag" or "soldbag",
                    indicating if the bag contains buy or sell transactions.

    Returns:
        The updated budget, which may be the same as the input budget if the last transaction's row index does not
        match the current row index.

    """
    # Determine the appropriate column index for the row index in the bag based on the bag_type.
    bag_row_idx = 2 if bag_type == "buybag" else 9
    # Check if there are any transactions in the bag.
    if bag.shape[0] > 0:
        # Compare the row index of the last transaction to the current row index.
        row_idx_last_tx = bag[-1, bag_row_idx]
        # Update the budget with the value from the last transaction in the bag.
        if row_idx_last_tx == row_idx:
            budget = bag[-1, 8]
    return budget


# cache needs to be off in case of a user defined trading strategy function
# @perf.jit_timer
@nb.njit(cache=False, parallel=False)
def iterate_market_history(
    params: dict,
    subset_bounds: tuple[int, int],
    asset_prices: np.ndarray,
    asset_prices_pct: np.ndarray,
    buy_signals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterate through the market history within a specified subset

    performing trading operations based on the given parameters.

    Params:
        params:            Dict containing simulation parameters.
        subset_bounds:     Tuple with the start and end indices of the current subset.
        asset_prices:      Array with the asset prices.
        asset_prices_pct:  Array with the asset prices percentage.
        buy_signals:       Array containing the buy signals.

    Returns:
        A tuple with two numpy arrays, soldbag and buybag, representing the transactions.
    """
    budget = params["budget"]
    buybag = np.empty((0, 9), type_float)
    soldbag = np.empty((0, 15), type_float)
    start_idx, end_idx = subset_bounds

    for row_idx, history_prices_row in enumerate(asset_prices[start_idx:end_idx]):
        # row_idx needs to get shifted by start_idx (start idx of the current subset iteration)
        row_idx += start_idx
        params["row_idx"] = row_idx

        soldbag, buybag = check_sell(params, buybag, soldbag, history_prices_row, budget)
        budget = update_budget_from_last_transaction(row_idx, budget, soldbag, "soldbag")

        buybag = check_buy(params, asset_prices_pct, buy_signals, buybag, history_prices_row, budget)
        budget = update_budget_from_last_transaction(row_idx, budget, buybag, "buybag")

    return soldbag, buybag


# cache needs to be off in case of a user defined trading strategy function
@nb.njit(cache=False, parallel=False)
def perform_trading_simulations(
    params: dict,
    asset_prices: np.ndarray,
    asset_prices_pct: np.ndarray,
    buy_signals: np.ndarray,
) -> tuple[np.int64, np.ndarray, np.ndarray]:
    """Perform trading simulations

    using the given parameters and market data: Iterate over all subsets
    of the provided market data and simulate trading within those subsets
    based on the specified trading strategy and parameters.

    Finally calculate the score, also gather trades history and buy log for
    the entire simulation.

    Params:
        params:            Dict containing simulation parameters,
                            such as trading strategy, investment amount, and trading fees.

        asset_prices:      Array containing the asset prices for each trading day.

        asset_prices_pct:  Array containing the asset price percentage changes
                            for each trading day.

        buy_signals:       Array containing the buy signals.

    Returns:
        A tuple with three elements:
            - Score: The mean profit across all subsets, adjusted by subtracting the
                        standard deviation of the profits.

            - Array containing the trading history across all subsets,
                including buy and sell events.

            - Array containing the buy log,
                which records the details of each buy event.
    """

    # Set the boundary for the market data subsets:
    # -> boundary is eqivalent to the length of the whole market data
    subset_idx_boundary = asset_prices.shape[0]

    # Sanitize the subset parameters to ensure they are within acceptable limits.
    _subset_size_increments, subset_amount = sim_utils.sanitize_subset_params(params, subset_idx_boundary)

    subset_start_idxs = sim_utils.get_subset_indices(
        params["moving_window_increments"], subset_idx_boundary, subset_amount, _subset_size_increments
    )

    # Initialize arrays to store profits and combined trading history
    profit_subset_list = np.empty(subset_amount, type_float)
    soldbag_all_subsets = np.empty((0, 15), type_float)

    # Iterate over the subsets
    for current_iter, subset_idx in enumerate(subset_start_idxs):
        subset_bounds = (subset_idx, subset_idx + _subset_size_increments)

        # Simulate trading by iterating over the market history of the subset
        soldbag, buybag = iterate_market_history(params, subset_bounds, asset_prices, asset_prices_pct, buy_signals)

        # Calculate the total profit for the current subset
        profit_subset_list[current_iter] = sim_utils.calc_metrics(soldbag)

        # Collect and store the trading history for the current subset
        soldbag_all_subsets = np.vstack((soldbag_all_subsets, soldbag))

    # Calculate the score for the entire simulation:
    # mean(subset profits) - std(subset profits)
    profit_total_std = np.std(profit_subset_list)
    profit_total = np.int64(np.mean(profit_subset_list) - profit_total_std)

    return profit_total, soldbag_all_subsets, buybag


def print_simulation_details(root: Tradeforce, asset_prices: pd.DataFrame, dataset_type: str) -> None:
    """Print simulation details.

    Params:
        root:         The Tradeforce instance containing the configuration.
        asset_prices: A DataFrame containing the asset prices.
    """

    log_dataset_type = "[Training]" if dataset_type == "train" else "[Validation]"

    history_begin = asset_prices.index.values.astype(int)[0]
    history_end = asset_prices.index.values.astype(int)[-1]

    history_delta = get_timedelta(history_end - history_begin, unit="ms")["datetime"]
    root.log.info(
        "%s Starting simulation | Timestamp %s to %s | Timeframe: %s",
        log_dataset_type,
        history_begin,
        history_end,
        history_delta,
    )
    if root.config.subset_amount > 1:
        root.log.info(
            "Simulation is split into %s subsets with increment_size = %s [%s days] each.",
            root.config.subset_amount,
            root.config._subset_size_increments,
            root.config.subset_size_days,
        )


def run(root: Tradeforce, dataset_type: str, train_val_split_idx: int) -> dict[str, int | np.ndarray]:
    """Run the trading simulation

    using the configuration provided in the Tradeforce instance. This function is the main entry point
    for simulations. It preprocesses the market history data, converts relevant data structures to
    numpy arrays for usage in Numba, and then performs the trading simulations to calculate the
    score, which is defined as:

        mean(profit subset) - std(profit subset).

    Params:
        root: The Tradeforce instance containing the necessary configuration and market history data.

    Returns:
        Dictionary containing the simulation results:
        - 'score'   (int):      The score calculated as: mean(profit subsets) - std(profit subsets).
        - 'trades'  (np.array): Array representing the trading history, including buy and sell events.
        - 'buy_log' (np.array): Array representing the buy log, containing the details of each buy event.
    """
    preprocess_result = pre_process(root.config, root.market_history, dataset_type, train_val_split_idx)
    sim_config = sim_utils.to_numba_dict(root.config.to_dict())

    asset_prices = preprocess_result.get("asset_prices", pd.DataFrame())
    asset_prices_pct = preprocess_result.get("asset_prices_pct", pd.DataFrame())
    buy_signals = preprocess_result.get("buy_signals", pd.DataFrame())

    print_simulation_details(root, asset_prices, dataset_type)

    np_asset_prices = asset_prices.to_numpy() if asset_prices is not None else None
    np_asset_prices_pct = asset_prices_pct.to_numpy() if asset_prices_pct is not None else None
    np_buy_signals = buy_signals.to_numpy() if buy_signals is not None else None

    score, trades_history, buy_log = perform_trading_simulations(
        sim_config, np_asset_prices, np_asset_prices_pct, np_buy_signals
    )

    suffix = "_val" if dataset_type == "val" else ""

    return {f"score{suffix}": score, f"trades{suffix}": trades_history, f"buy_log{suffix}": buy_log}


def run_train_val_split(root: Tradeforce) -> dict[str, int | np.ndarray]:
    n_market_history = root.market_history.get_history_size()

    train_val_split_idx = int(n_market_history * root.config.train_val_split_ratio)

    # Cache the original subset parameters
    cache_subset_amount = root.config.subset_amount
    cache_subset_size_days = root.config.subset_size_days
    cache_subset_size_increments = root.config._subset_size_increments

    sim_result_trainset = run(root, "train", train_val_split_idx)

    # Do not use subsets for validation set
    root.config.subset_amount = -1
    root.config.subset_size_days = -1
    root.config._subset_size_increments = -1

    sim_result_valset = run(root, "val", train_val_split_idx)

    # Restore the original subset parameters
    root.config.subset_amount = cache_subset_amount
    root.config.subset_size_days = cache_subset_size_days
    root.config._subset_size_increments = cache_subset_size_increments

    # Merge the results from the train and val sets
    sim_result = {**sim_result_trainset, **sim_result_valset}

    return sim_result
