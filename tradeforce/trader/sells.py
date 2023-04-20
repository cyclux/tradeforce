""" trader/sells.py

Module: tradeforce.trader.sells
-------------------------------

Provides functions to sell assets for the Tradeforce Trader.

Calculates and checks for profitable sell conditions based on the
current market situation, the user's portfolio, and the time since
the asset was bought. It also handles the simulation of selling
assets and updating budgets based on the confirmed sell orders.

The module interacts with the main Tradeforce instance, the trader,
and the exchange API to submit sell orders, update order information,
and manage the user's portfolio. It relies on the tradeforce.utils
module for helper functions related to fees and symbol conversions.

Main Functions:

    check_sell_options(root: Tradeforce) -> list[dict]:
        Check for sell options in the current portfolio.

    sell_assets(root: Tradeforce, sell_options: list[dict]):
        Sell assets based on the provided sell options.

    submit_sell_order(root: Tradeforce, open_order: dict):
        Submit a sell order based on the given open order.

    sell_confirmed(root: Tradeforce, sell_order: dict):
        Process the confirmed sell order and update
            the open and closed orders accordingly.

    update_budget(root: Tradeforce, closed_order: dict):
        Update the budget based on the closed order.

"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from tradeforce.utils import calc_fee, convert_symbol_from_exchange

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import Tradeforce


def _get_latest_prices(root: Tradeforce) -> tuple[int, dict]:
    """Get the latest prices for all symbols from the market history.

    Params:
        root: The main Tradeforce instance providing
        access to the market_history module.

    Returns:
        A tuple containing the timestamp and a dictionary of the latest prices.
    """
    df_latest_prices = root.market_history.get_market_history(latest_candle=True, metrics=["c"], uniform_cols=True)
    timestamp = df_latest_prices.index.values[0]
    latest_prices = df_latest_prices.to_dict("records")[0]
    return timestamp, latest_prices


def _should_sell_simulation(root: Tradeforce, price_current: float, price_profit: float, ok_to_sell: bool) -> bool:
    """Determine if the asset should be sold during a simulation.

    "1.2" is the maximum allowed profit ratio for a trade in a simulation.
    This prevents the simulation from selling assets for a profit that is not realistic.

    Params:
        root:          The main Tradeforce instance.
        price_current: The current price of the asset.
        price_profit:  The target price for profit.
        ok_to_sell:    Whether the asset meets the conditions to be sold.

    Returns:
        True if the asset should be sold, False otherwise.
    """
    return (price_current >= price_profit or ok_to_sell) and (price_current / price_profit <= 1.2)


def _create_sell_option(root: Tradeforce, open_order: dict, price_current: float) -> dict:
    """Create a dictionary containing the sell option information.

    Params:
        root:          The main Tradeforce instance.
        open_order:    The open order information.
        price_current: The current price of the asset.

    Returns:
        Dict containing the sell option information.
    """
    sell_option = {
        "asset": open_order["asset"],
        "price_sell": price_current,
    }
    if not root.config.is_sim:
        sell_option.update(
            {
                "gid": open_order["gid"],
                "buy_order_id": open_order["buy_order_id"],
            }
        )
    return sell_option


def _get_current_profit_ratio(price_current: float, price_buy: float) -> float:
    """Calculate the current profit ratio.

    Params:
        price_current: The current price of the asset.
        price_buy:     The initial buying price of the asset.

    Returns:
        The current profit ratio of the given asset.
    """
    return np.round(price_current / price_buy, 2)


def _get_increments_since_buy(timestamp: int, timestamp_buy: int, candle_interval: str) -> int:
    """Calculate the increments since the asset was bought.

    The amount of increments is variable and depends
    on the candle_interval of the market history.

    Params:
        timestamp:     The current timestamp.
        timestamp_buy: The timestamp when the asset was bought.

    Returns:
        The time since the asset was bought in candle intervals
            of the market history (candle_interval).
    """
    candle_interval_in_minutes = pd.Timedelta(candle_interval).value / 10**9 / 60
    return int((timestamp - timestamp_buy) / 1000 / 60 / candle_interval_in_minutes)


def _is_ok_to_sell(root: Tradeforce, time_since_buy: int, current_profit_ratio: float) -> bool:
    """Determine selling the asset based on the time_since_buy and current_profit_ratio.

    Params:
        root:                 The main Tradeforce instance.
        time_since_buy:       The time since the asset was bought.
        current_profit_ratio: The current profit ratio.

    Returns:
        True if it's okay to sell the asset, False otherwise.
    """
    return (
        time_since_buy > root.config._hold_time_increments
        and current_profit_ratio >= root.config.profit_factor_target_min
    )


def _process_open_order(root: Tradeforce, open_order: dict, latest_prices: dict, timestamp: int) -> dict | None:
    """Process an open order and return a sell option if conditions are met.

    Params:
        root:          The main Tradeforce instance.
        open_order:    The open order information.
        latest_prices: Dict containing the latest prices for all symbols.
        timestamp:     The current timestamp.

    Returns:
        Dict containing the sell option information
            if conditions are met, None otherwise.
    """
    symbol = open_order["asset"]
    price_current = latest_prices.get(symbol, 0)
    price_buy = open_order["price_buy"]
    price_profit = open_order["price_profit"]

    if np.isnan(price_current):
        price_current = 0.0

    current_profit_ratio = _get_current_profit_ratio(price_current, price_buy)
    time_since_buy = _get_increments_since_buy(timestamp, open_order["timestamp_buy"], root.config.candle_interval)
    ok_to_sell = _is_ok_to_sell(root, time_since_buy, current_profit_ratio)

    if root.config.is_sim:
        if _should_sell_simulation(root, price_current, price_profit, ok_to_sell):
            price_current = min(price_current, price_profit)
            return _create_sell_option(root, open_order, price_current)
    elif ok_to_sell:
        return _create_sell_option(root, open_order, price_current)

    return None


def check_sell_options(root: Tradeforce) -> list[dict]:
    """Check for sell options in the current portfolio.

    Params:
        root: The main Tradeforce instance providing access
        to the config, trader, logger or any other module.

    Returns:
        A list of sell options.
    """
    sell_options = []
    portfolio_performance = {}

    timestamp, latest_prices = _get_latest_prices(root)

    open_orders = root.trader.get_all_open_orders()

    for open_order in open_orders:
        sell_option = _process_open_order(root, open_order, latest_prices, timestamp)
        if sell_option is not None:
            sell_options.append(sell_option)
            symbol = open_order["asset"]
            price_current = latest_prices.get(symbol, 0)
            price_buy = open_order["price_buy"]
            portfolio_performance[symbol] = _get_current_profit_ratio(price_current, price_buy)

    if not root.config.is_sim:
        root.log.info("Current portfolio performance: %s", portfolio_performance)

    return sell_options


def _create_sell_order_edit(open_order: dict, sell_option: dict) -> dict:
    """Create a sell order dict
        by editing the given open order with the price_sell of the sell_option.

    Params:
        open_order:  The open order dictionary.
        sell_option: The sell option dictionary.

    Returns:
        Dict containing the sell order items.
    """
    volatility_buffer = 0.00000005
    return {
        "sell_order_id": open_order["sell_order_id"],
        "gid": open_order["gid"],
        "asset": open_order["asset"],
        "price": sell_option["price_sell"],
        "amount": open_order["buy_volume_asset"] - volatility_buffer,
    }


async def _process_sell_option(root: Tradeforce, open_order: dict, sell_option: dict) -> bool:
    """Process a sell option and update the sell order if necessary.

    Params:
        root:        The main Tradeforce instance.
        open_order:  The open order dictionary.
        sell_option: The sell option dictionary.

    Returns:
        True if the sell order was successfully updated, False otherwise.
    """
    sell_order = _create_sell_order_edit(open_order, sell_option)
    order_result_ok = await root.exchange_api.edit_order("sell", sell_order)

    if order_result_ok:
        root.log.info(
            "Sell price of %s has been changed from %s to %s",
            sell_order["asset"],
            open_order["price_buy"],
            sell_order["price"],
        )
    return order_result_ok


async def sell_assets(root: Tradeforce, sell_options: list[dict]) -> None:
    """Sell assets based on the provided sell options.

    Params:
        root:         The main Tradeforce instance.
        sell_options: A list of sell options.
    """
    for sell_option in sell_options:
        open_order = root.trader.get_open_order(asset=sell_option)
        if not open_order:
            continue

        if root.config.is_sim:
            sell_confirmed(root, sell_option)
        else:
            await _process_sell_option(root, open_order[0], sell_option)


def _create_sell_order(open_order: dict) -> dict:
    """Create a dictionary containing the sell order items.

    Params:
        open_order: The open order.

    Returns:
        Dict containing the sell order items.
    """
    volatility_buffer = 0.00000005
    return {
        "asset": open_order["asset"],
        "price": open_order["price_profit"],
        "amount": open_order["buy_volume_asset"] - volatility_buffer,
        "gid": open_order["gid"],
    }


async def submit_sell_order(root: Tradeforce, open_order: dict) -> None:
    """Submit a sell order based on the given open order.

    Params:
        root:       The main Tradeforce instance.
        open_order: The open order dictionary.

    Raises:
        If the sell order execution fails, logs an error message with the sell order details.
    """
    sell_order = _create_sell_order(open_order)
    exchange_result_ok = await root.exchange_api.order("sell", sell_order)

    if not exchange_result_ok:
        root.log.error("Sell order execution failed! -> %s", str(sell_order))


def _create_closed_order(root: Tradeforce, open_order: dict, sell_order: dict) -> dict:
    """Create a closed order dictionary based on the open order and sell order.

    Params:
        root:       The main Tradeforce instance.
        open_order: The open order dictionary.
        sell_order: The sell order dictionary.

    Returns:
        Dict containing the closed order details.
    """
    closed_order = open_order.copy()
    closed_order["timestamp_sell"] = sell_order["mts_update"]
    closed_order["price_sell"] = sell_order["price_avg"]
    sell_volume_asset = abs(sell_order["amount_orig"])
    sell_volume_asset_incl_fee, _, closed_order["sell_fee_fiat"] = calc_fee(
        root.config, sell_volume_asset, sell_order["price_avg"], order_type="sell"
    )
    sell_volume_fiat_incl_fee = sell_volume_asset_incl_fee * sell_order["price_avg"]
    closed_order["sell_volume_fiat"] = np.round(sell_volume_fiat_incl_fee - closed_order["sell_fee_fiat"], 3)
    closed_order["profit_fiat"] = closed_order["sell_volume_fiat"] - closed_order["amount_invest_per_asset"]
    return closed_order


def update_budget(root: Tradeforce, closed_order: dict) -> None:
    """Update the budget based on the closed order.

    Params:
        root:         The main Tradeforce instance.
        closed_order: The closed order dictionary.
    """
    new_budget = float(np.round(root.config.budget + closed_order["sell_volume_fiat"], 2))
    root.backend.update_status({"budget": new_budget})


def sell_confirmed(root: Tradeforce, sell_order: dict) -> None:
    """Process the confirmed sell order and update the open and closed orders accordingly.

    Params:
        root:       The main Tradeforce instance.
        sell_order: The sell order dictionary.
    """
    root.log.debug("sell_order confirmed: %s", sell_order)
    asset_symbol = convert_symbol_from_exchange(sell_order["symbol"])[0]
    open_order = root.trader.get_open_order(asset={"asset": asset_symbol, "gid": sell_order["gid"]})
    if open_order:
        closed_order = _create_closed_order(root, open_order[0], sell_order)
        root.trader.new_order(closed_order, "closed_orders")
        root.trader.del_order(open_order[0], "open_orders")

        if root.config.is_sim:
            update_budget(root, closed_order)
    else:
        root.log.error("Could not find order to sell: %s", sell_order)
