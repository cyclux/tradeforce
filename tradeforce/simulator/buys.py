""" simulator/buys.py

Module: tradeforce.simulator.buys
---------------------------------

Makes use of Numba for just-in-time compilation to gain major performance improvement in simulations.

Provide functions for managing the buying of assets in the TradeForce simulator.
It includes functions for evaluating assets for purchase based on the given strategy,
checking if the maximum number of buys per asset has been reached, and determining
whether the total investment cap has been reached.

Main Functions:

    buy_asset:                     Calculates the parameters for buying an asset and updates the budget.
    check_buy:                     Determines which assets to buy based on the given strategy
                                    and updates the buybag.
    _is_max_buy_per_asset_reached: Checks if the maximum allowed number of buys
                                    for a given asset has been reached.
    _is_investment_cap_reached:    Checks if the total investment, including fees,
                                    has reached the specified investment cap.

"""

import numpy as np
import numba as nb  # type: ignore
import tradeforce.simulator.utils as sim_utils
import tradeforce.simulator.default_strategies as strategies

NB_PARALLEL = False
NB_CACHE = True


@nb.njit(cache=NB_CACHE, parallel=NB_PARALLEL)
def buy_asset(
    buy_option_idx: np.float64,
    row_idx: np.float64,
    price_current: np.float64,
    profit_factor_target: np.float64,
    amount_invest_per_asset: np.float64,
    maker_fee: np.float64,
    taker_fee: np.float64,
    budget: np.float64,
) -> tuple[np.ndarray, np.float64]:
    """Calculate the parameters for buying an asset and update the budget.

    Compute the price and amount invested in the asset, including fees, and update
    the remaining budget. Then return an array containing the relevant parameters for the
    purchased asset and the updated budget.

    Params:
        buy_option_idx: A float representing the index of the asset buy option.
        row_idx: A float representing the index of the current row in the market history.
        price_current: A float representing the current price of the asset.
        profit_factor_target: A float representing the target profit factor for the asset.
        amount_invest_per_asset: A float representing the amount to invest in the asset.
        maker_fee: A float representing the maker fee for the trade.
        taker_fee: A float representing the taker fee for the trade.
        budget: A float representing the current available budget.

    Returns:
        A tuple containing a numpy array with the parameters for the purchased asset and the updated budget.
    """
    price_profit = price_current * profit_factor_target
    amount_invest_asset = amount_invest_per_asset / price_current
    amount_invest_asset_incl_fee, _, amount_fee_buy_fiat = sim_utils.calc_fee(
        amount_invest_asset, maker_fee, taker_fee, price_current, "buy"
    )
    amount_invest_per_asset_incl_fee = amount_invest_per_asset - amount_fee_buy_fiat
    budget -= amount_invest_per_asset
    bought_asset_params = np.array(
        [
            [
                buy_option_idx,
                1.0,  # placeholder
                row_idx,
                price_current,
                price_profit,
                amount_invest_per_asset_incl_fee,
                amount_invest_asset_incl_fee,
                amount_fee_buy_fiat,
                budget,
            ]
        ]
    )
    return bought_asset_params, budget


# cache needs to be off in case of a user defined trading strategy function
@nb.njit(cache=False, parallel=NB_PARALLEL)
def _is_max_buy_per_asset_reached(buy_option_idx: np.float64, buybag: np.ndarray, max_buy_per_asset: int) -> bool:
    """Check if the maximum allowed number of buys for a given asset has been reached.

    Params:
        buy_option_idx: A float representing the index of the asset being considered for purchase.
        buybag: Array containing the purchased assets and their parameters.
        max_buy_per_asset: An integer representing the maximum allowed number of buys per asset.

    Returns:
        A boolean indicating whether the maximum allowed number of buys for the given asset has been reached.
    """
    if buy_option_idx in buybag[:, 0:1]:
        buybag_idx = np.where(buybag[:, 0] == buy_option_idx)[0]
        asset_buy_prices = buybag[buybag_idx, 3]
        if len(asset_buy_prices) >= max_buy_per_asset:
            return True
    return False


@nb.njit(cache=False, parallel=NB_PARALLEL, fastmath=True)
def _is_investment_cap_reached(buybag: np.ndarray, investment_cap: np.float64) -> bool:
    """Check if the total investment, including fees, has reached the specified investment cap.

    Params:
        buybag: Array containing the purchased assets and their parameters.
        investment_cap: A float representing the maximum allowed investment.

    Returns:
        A boolean indicating whether the total investment has reached the investment cap.
    """
    investment_fiat_incl_fee = buybag[:, 5:6]
    fee_fiat = buybag[:, 7:8]
    investment_total = np.sum(investment_fiat_incl_fee + fee_fiat)
    return investment_total >= investment_cap > 0


@nb.njit(cache=False, parallel=NB_PARALLEL)
def check_buy(
    params: dict,
    df_asset_prices_pct: np.ndarray,
    df_buy_signals: np.ndarray,
    buybag: np.ndarray,
    history_prices_row: np.ndarray,
    budget: np.float64,
) -> np.ndarray:
    """Determine which assets to buy based on the given strategy and update the buybag.

    Evaluate assets for purchase using the specified strategy: check whether the
    budget and investment cap allow for purchasing, and update the buybag accordingly.

    Params:
        params: Dict containing strategy parameters and other settings.
        df_asset_prices_pct: Array containing asset price percentage changes.
        df_buy_signals: Array containing asset performance metrics.
        buybag: Array containing the purchased assets and their parameters.
        history_prices_row: Array containing the historical prices of assets.
        budget: A float representing the current available budget.

    Returns:
        An updated numpy array containing the purchased assets and their parameters.
    """
    list_buy_options = strategies.buy_strategy(params, df_asset_prices_pct, df_buy_signals)

    for buy_option_idx in list_buy_options:
        price_current = history_prices_row[buy_option_idx]

        if _is_max_buy_per_asset_reached(buy_option_idx, buybag, params["max_buy_per_asset"]):
            continue

        if _is_investment_cap_reached(buybag, params["investment_cap"]):
            continue

        if budget >= params["amount_invest_per_asset"]:
            bought_asset_params, budget = buy_asset(
                buy_option_idx,
                params["row_idx"],
                price_current,
                params["profit_factor_target"],
                params["amount_invest_per_asset"],
                params["maker_fee"],
                params["taker_fee"],
                budget,
            )
            buybag = np.append(buybag, bought_asset_params, axis=0)

    return buybag
