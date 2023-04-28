""" simulator/sells.py

Module: tradeforce.simulator.sells
----------------------------------

Provide Numba optimized functions for managing the selling of assets
in the TradeForce simulator. This includes functions for evaluating
assets for selling based on the given strategy, calculating
sell-related parameters for each asset, and executing sell orders.

Main Functions:

    check_sell:                 Checks and executes sell orders based
                                    on the provided sell strategy for
                                    the assets in the buybag.

    sell_asset:                 Executes sell orders for the specified
                                    assets and updates the soldbag with
                                    the sold assets and their parameters.

    _calculate_sell_parameters: Calculates the selling-related parameters
                                    for each asset being sold.

"""

import numpy as np
import numba as nb  # type: ignore

from tradeforce.simulator.utils import calc_fee
import tradeforce.simulator.default_strategies as strategies

NB_PARALLEL = False
NB_CACHE = True

type_float = nb.typeof(0.1)


@nb.njit(cache=NB_CACHE, parallel=False, fastmath=True)
def _calculate_sell_parameters(
    params: dict,
    assets_to_sell: np.ndarray,
    prices_current: np.ndarray,
    budget: np.float64,
) -> tuple[np.ndarray, np.float64, np.float64, np.ndarray, np.ndarray]:
    """Calculate the selling-related parameters for each asset being sold.

    Including the fiat value including fees, profit in fiat,
    sold asset amount including fees, and the fees in fiat.

    Params:
        params: Dict containing the parameters for the sell strategy.

        assets_to_sell: Array containing the assets to be sold,
                            including their buy parameters.

        prices_current: Array containing the current prices of the
                            assets being sold.

        budget: A float representing the current budget.

    Returns:
        A tuple containing the following numpy arrays:
            - amount_sold_fiat_incl_fee: The fiat value of the sold assets including fees.
            - amount_profit_fiat: The profit in fiat obtained from selling the assets.
            - new_budget: The updated budget after selling the assets.
            - amount_sold_asset_incl_fee: The sold asset amount including fees.
            - amount_fee_sell_fiat: The fees in fiat for selling the assets.
    """
    amounts_invest_asset = assets_to_sell[:, 6:7]
    amount_sold_asset_incl_fee, _, amount_fee_sell_fiat = calc_fee(
        amounts_invest_asset, params["maker_fee"], params["taker_fee"], prices_current, "sell"
    )
    amount_sold_fiat_incl_fee = amount_sold_asset_incl_fee * prices_current
    amount_profit_fiat = amount_sold_fiat_incl_fee - params["amount_invest_per_asset"]

    new_budget = np.sum(amount_sold_fiat_incl_fee) + budget

    return amount_sold_fiat_incl_fee, amount_profit_fiat, new_budget, amount_sold_asset_incl_fee, amount_fee_sell_fiat


@nb.njit(cache=NB_CACHE, parallel=False)
def sell_asset(
    params: dict,
    assets_to_sell: np.ndarray,
    soldbag: np.ndarray,
    prices_current: np.ndarray,
    budget: np.float64,
) -> np.ndarray:
    """Execute sell orders for the specified assets
        and update the soldbag with the sold assets and their parameters.

    Calculate the selling-related parameters for each asset:
    The fiat value including fees, profit in fiat, sold asset amount including fees,
    and the fees in fiat. Finally append the sold assets and their
    parameters to the soldbag.

    Params:
        params:         Dict containing the parameters for the sell strategy.
        assets_to_sell: Array containing the assets to be sold,
                        including their buy parameters.
        soldbag:        Array containing the sold assets and their parameters,
                            such as sell timestamps, fees, and profits.
        prices_current: Array containing the current prices of the assets being sold.
        budget:         A float representing the current budget.

    Returns:
        Array containing the updated soldbag with the sold assets
        and their parameters, such as sell timestamps, fees, and profits.
    """

    amount_assets_sell = assets_to_sell.shape[0]
    sold_assets = np.empty((amount_assets_sell, 15), type_float)

    (
        amount_sold_fiat_incl_fee,
        amount_profit_fiat,
        new_budget,
        amount_sold_asset_incl_fee,
        amount_fee_sell_fiat,
    ) = _calculate_sell_parameters(params, assets_to_sell, prices_current, budget)

    new_budget_reshaped = np.broadcast_to(new_budget, (amount_assets_sell, 1))
    assets_to_sell[:, 8:9] = new_budget_reshaped

    row_idx_sold = np.broadcast_to(params["row_idx"], (amount_assets_sell, 1))
    sold_assets[:, 0:9] = assets_to_sell
    sold_assets[:, 9:10] = row_idx_sold
    sold_assets[:, 10:11] = prices_current
    sold_assets[:, 11:12] = amount_sold_fiat_incl_fee
    sold_assets[:, 12:13] = amount_sold_asset_incl_fee
    sold_assets[:, 13:14] = amount_fee_sell_fiat
    sold_assets[:, 14:15] = amount_profit_fiat

    soldbag = np.append(soldbag, sold_assets, axis=0)
    return soldbag


# cache needs to be off in case of a user defined trading strategy function
@nb.njit(cache=False, parallel=False, fastmath=True)
def check_sell(
    params: dict, buybag: np.ndarray, soldbag: np.ndarray, history_prices_row: np.ndarray, budget: np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """Check and execute sell orders
        based on the provided sell strategy for the assets in the buybag.

    First check if there are assets in the buybag to sell. If there are,
    evaluate the sell strategy to identify the assets to sell and their
    current prices. Execute sell orders for the identified assets,
    update the soldbag with the sold assets and their parameters.
    Finally remove the sold assets from the buybag.

    Params:
        params:             Dict containing the parameters
                                for the sell strategy.

        buybag:             Array containing the purchased
                                assets and their parameters, such
                                as buy timestamps and fees.

        soldbag:            Array containing the sold assets
                                and their parameters, such as sell
                                timestamps, fees, and profits.

        history_prices_row: Array containing the current
                                row of historical prices.

        budget:             A float representing the current budget.

    Returns:
        A tuple containing the updated soldbag and buybag after executing
            sell orders. The soldbag will include the sold assets and their
            parameters, while the buybag will have the sold assets removed.
    """
    amount_buy_orders = buybag.shape[0]
    if amount_buy_orders < 1:
        return soldbag, buybag

    sell_assets, prices_current = strategies.sell_strategy(params, buybag, history_prices_row)

    assets_to_sell = buybag[sell_assets, :]
    if assets_to_sell.shape[0] > 0:
        soldbag = sell_asset(
            params,
            assets_to_sell,
            soldbag,
            prices_current[sell_assets, :],
            budget,
        )
        buybag = buybag[~sell_assets, :]
    return soldbag, buybag
