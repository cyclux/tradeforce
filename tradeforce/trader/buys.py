""" trader/buys.py

Module: tradeforce.trader.buys
------------------------------

Provides functions to buy assets for the Tradeforce Trader.
The main function in this module is buy_assets, which takes a list
of buy option dictionaries and executes buy orders for each asset in the list.
It handles skipping assets based on exclusion lists and existing open orders,
adjusting the buy amount to meet the minimum order size, processing buy options,
logging a summary of the buying process, and handling confirmed buy orders.

Other functions in this module include:

    buy_confirmed: Handles a confirmed buy order by building an open order,
        adding it to the trader, and submitting a sell order which includes the profit factor.

    _should_skip_asset: Determines if an asset should be skipped based on
        exclusion lists and existing open orders.

    _adjust_buy_amount_asset: Adjusts the buy amount of assetcurrency
        to meet the minimum order size.

    _process_buy_option: Processes a single buy option for a given asset.

    _log_summary: Logs a summary of the buying process, including assets that were
        out of funds to buy and assets that reached the maximum buying amount.

    _get_buy_volume_asset: Gets the buy volume of an asset.

    _build_open_order: Builds an open order dictionary based on the provided buy order.

"""

from __future__ import annotations
from asyncio import sleep as asyncio_sleep
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from tradeforce.utils import convert_symbol_from_exchange
from tradeforce.market.metrics import calc_buy_signals
from tradeforce.trader.sells import submit_sell_order

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import Tradeforce
    from bfxapi.models.order import Order  # type: ignore


def _get_significant_digits(num: float, digits: int) -> float:
    """Get the significant digits of a number rounded to a specific number of digits.

    Significant digits are the digits that are not zero.
    e.g. 0.0001234 has 4 significant digits.
         and 0.0001234 rounded to 2 significant digits is 0.00012
         and 1234.567800 has 8 significant digits.

    Params:
        num:    The input number.
        digits: The number of significant digits to round to.

    Returns:
        The number rounded to the specified number of significant digits.
    """
    return round(num, digits - int(np.floor(np.log10(abs(num)))) - 1)


def _get_latest_prices_and_timestamp(root: Tradeforce) -> tuple[dict, int]:
    """Get the latest prices and their corresponding timestamp.

    Params:
        root: The main Tradeforce instance providing access
        to the market_history module.

    Returns:
        A tuple containing the latest prices as a dictionary
        and the corresponding timestamp.
    """
    df_latest_prices = root.market_history.get_market_history(latest_candle=True, metrics=["o"], uniform_cols=True)
    latest_prices = df_latest_prices.to_dict("records")[0]
    timestamp = df_latest_prices.index.values[0]
    return latest_prices, timestamp


def _filter_buy_signals(buy_signals: pd.Series, buy_signal_score: float, buy_signal_boundary: float) -> pd.Series:
    """Filter price performance (signal score)
    based on buy opportunity factors and boundaries.

    Params:
        buy_performance:          The buy performance series.
        buy_signal_score:   The buy opportunity factor.
        buy_signal_boundary: The buy opportunity boundary.

    Returns:
        A filtered signal score series.
    """
    buy_signal_score_min = buy_signal_score - buy_signal_boundary
    buy_signal_score_max = buy_signal_score + buy_signal_boundary
    buy_condition = (buy_signals >= buy_signal_score_min) & (buy_signals <= buy_signal_score_max)
    return buy_signals[buy_condition]


def _filter_and_sort_buy_signals(
    df_buy_signals: pd.DataFrame, buy_signal_preference: int, buy_signal_score: float
) -> pd.DataFrame:
    """Filter and sort buy signals based on 'signal preference' and signal score.

    Performance means average profit per unit of time.
    Performance preference means that the buy options are sorted by performance in ascending or descending order.
    buy_signal_preference == -1 means that those assets with the lowest performance are preferred.
    buy_signal_preference == 1 means that those assets with the highest performance are preferred.
    buy_signal_preference == 0 means that those assets closest to the signal score are preferred.

    Params:
        df_buy_options:         The dataframe containing buy options.
        buy_signal_preference:     Performance preference (-1 for low, 0 for neutral, 1 for high).
        buy_signal_score: The buy opportunity factor.

    Returns:
        A filtered and sorted dataframe of buy options.
    """
    if buy_signal_preference == -1:
        df_buy_signals = df_buy_signals.sort_values(by="perf", ascending=True)
    elif buy_signal_preference == 1:
        df_buy_signals = df_buy_signals.sort_values(by="perf", ascending=False)
    elif buy_signal_preference == 0:
        df_buy_signals.loc[:, "perf"] = np.absolute(df_buy_signals["perf"] - buy_signal_score)
        df_buy_signals = df_buy_signals.sort_values(by="perf")

    return df_buy_signals


def _log_buy_options(root: Tradeforce, buy_options: list[dict]) -> None:
    """Log the buy options.

    Params:
        root:        The main Tradeforce instance.
        buy_options: The list of buy options as dictionaries.
    """
    if buy_options:
        buy_options_print = [
            f"{buy_option['asset']} [perf:{np.round(buy_option['perf'], 2)}, price: {buy_option['price']}]"
            for buy_option in buy_options
        ]
        root.log.info(
            "%s potential asset%s to buy: %s",
            len(buy_options),
            "s" if len(buy_options) > 1 else "",
            buy_options_print,
        )
    else:
        root.log.info("Currently no potential assets to buy")


def check_buy_options(root: Tradeforce, latest_prices: dict | None = None, timestamp: int | None = None) -> list[dict]:
    """Check buy options / buy signal based on the given latest prices and timestamp.

    Params:
        root:          The main Tradeforce instance.
        latest_prices: The latest prices of all assets as a dictionary.
        timestamp:     The timestamp of the latest prices.

    Returns:
        A list of buy options as dictionaries.
    """
    buy_options = []

    if latest_prices is None or timestamp is None:
        latest_prices, timestamp = _get_latest_prices_and_timestamp(root)

    buy_signals = calc_buy_signals(
        root, moving_window_increments=root.config.moving_window_increments, timestamp=timestamp
    )

    if buy_signals is not None:
        buy_signals = _filter_buy_signals(buy_signals, root.config.buy_signal_score, root.config.buy_signal_boundary)
        df_buy_options = pd.DataFrame({"perf": buy_signals, "price": latest_prices}).dropna()
        df_buy_options = _filter_and_sort_buy_signals(
            df_buy_options, root.config.buy_signal_preference, root.config.buy_signal_score
        )
        df_buy_options.reset_index(names=["asset"], inplace=True)
        buy_options = df_buy_options.to_dict("records")

    _log_buy_options(root, buy_options)

    return buy_options


async def _update_sim_budget(root: Tradeforce) -> None:
    """Update the simulated budget.

    Params:
        root: The main Tradeforce instance providing
        access to the config the backend.
    """

    new_budget = float(np.round(root.config.budget - root.config.amount_invest_per_asset, 2))
    root.backend.update_status({"budget": new_budget})


async def _execute_buy_order(root: Tradeforce, buy_order: dict) -> bool:
    """Execute a buy order.
        gid is the global id of the buy order.

    Params:
        root:      The main Tradeforce instance.
        buy_order: The buy order as a dictionary.

    Returns:
        True if the buy order execution was successful, False otherwise.
    """
    exchange_result_ok = await root.exchange_api.order("buy", buy_order)
    root.trader.gid += 1
    root.backend.update_status({"gid": root.trader.gid})

    if not exchange_result_ok:
        root.log.error("Buy order execution failed! -> %s", buy_order)
        return False

    return True


async def _process_buy_order(root: Tradeforce, asset_symbol: str, price: float, buy_amount_asset: float) -> bool:
    """Process a buy order for a given asset symbol
        or price, and buy amount in asset.

    Params:
        root:              The main Tradeforce instance.
        asset_symbol:      The asset symbol.
        price:             The price of the asset.
        buy_amount_asset: The amount of the asset to buy in asset.

    Returns:
        True if the buy order was successful, False otherwise.
    """
    buy_order = {
        "asset": asset_symbol,
        "gid": root.trader.gid,
        "price": price,
        "amount": buy_amount_asset,
    }
    root.log.info("Executing buy order: %s", buy_order)

    if root.config.is_sim:
        await _update_sim_budget(root)
        return True
    else:
        return await _execute_buy_order(root, buy_order)


async def _should_skip_asset(root: Tradeforce, asset_symbol: str, asset: dict) -> bool:
    """Determines if the asset should be skipped
        based on exclusion list and existing open orders.

    Params:
        root:         Tradeforce main instance.
        asset_symbol: The symbol of the asset.
        asset:        The asset dictionary.

    Returns:
        True if the asset should be skipped, False otherwise.
    """
    if asset_symbol in root.config.assets_excluded:
        root.log.info("Asset on blacklist. Will not buy %s", asset)
        return True

    asset_open_orders = root.trader.get_open_order(asset=asset)
    if asset_open_orders:
        return True

    return False


async def _adjust_buy_amount_asset(root: Tradeforce, asset_symbol: str, buy_amount_asset: float) -> float:
    """Adjusts the buy amount of assetcurrency to meet the minimum order size.
        Adds 2% to the minimum order size to account for fees.

    Params:
        root:              Tradeforce main instance.
        asset_symbol:      The symbol of the asset.
        buy_amount_asset: The initial buy amount of assetcurrency.

    Returns:
        The adjusted buy amount of assetcurrency.
    """
    min_order_size = root.trader.min_order_sizes.get(asset_symbol, 0)
    if min_order_size > buy_amount_asset:
        root.log.info(
            "Adapting buy_amount_asset (%s) of %s to min_order_size (%s).",
            buy_amount_asset,
            asset_symbol,
            min_order_size,
        )
        buy_amount_asset = min_order_size * 1.02

    return buy_amount_asset


async def _process_buy_option(root: Tradeforce, asset: dict) -> tuple[bool, bool]:
    """Processes a single buy option for a given asset.

    Params:
        root:  Tradeforce main instance.
        asset: The asset dictionary containing the asset symbol and price.

    Returns:
        A tuple containing two boolean values:
            1. True if the maximum amount of the asset has been bought, False otherwise.
            2. True if there are insufficient funds to buy the asset, False otherwise.
    """

    asset_symbol = asset["asset"]
    if await _should_skip_asset(root, asset_symbol, asset):
        return False, False

    asset["price"] *= 1.015
    buy_amount_asset = _get_significant_digits(root.config.amount_invest_per_asset / asset["price"], 9)
    buy_amount_asset = await _adjust_buy_amount_asset(root, asset_symbol, buy_amount_asset)

    if root.config.budget < root.config.amount_invest_per_asset:
        return False, True

    successful_execution = await _process_buy_order(root, asset_symbol, asset["price"], buy_amount_asset)

    return False, not successful_execution


async def buy_assets(root: Tradeforce, buy_options: list[dict]) -> None:
    """Buys assets based on the provided buy options.

    max_bought and out_of_funds are used to determine
    if the maximum amount of an asset has been bought
    or if there are insufficient funds to buy the asset.

    compensate_rate_limit prevents the rate limit of the exchange API from being exceeded

    Params:
        root:        Tradeforce main instance.
        buy_options: A list of buy option dictionaries.
    """
    compensate_rate_limit = bool(len(buy_options) > 9)
    assets_out_of_funds_to_buy = []
    assets_max_amount_bought = []

    for asset in buy_options:
        max_bought, out_of_funds = await _process_buy_option(root, asset)
        if max_bought:
            assets_max_amount_bought.append(asset["asset"])
        if out_of_funds:
            assets_out_of_funds_to_buy.append(asset["asset"])

        if compensate_rate_limit:
            await asyncio_sleep(0.8)

    _log_summary(root, assets_out_of_funds_to_buy, assets_max_amount_bought)


def _log_summary(root: Tradeforce, assets_out_of_funds_to_buy: list[str], assets_max_amount_bought: list[str]) -> None:
    """Logs a summary of the buying process,

    including assets that were out of funds to buy and assets that reached
    the maximum buying amount.

    Params:
        root:                       Tradeforce main instance.
        assets_out_of_funds_to_buy: A list of asset symbols that were out of funds to buy.
        assets_max_amount_bought:   A list of asset symbols that reached the maximum buying amount.
    """
    plural_assets_out_of_funds = "s" if len(assets_out_of_funds_to_buy) > 1 else ""
    plural_assets_max_amount_bought = "s" if len(assets_max_amount_bought) > 1 else ""

    if assets_out_of_funds_to_buy:
        root.log.info(
            "%s asset%s out of funds to buy ($%s < $%s): %s",
            len(assets_out_of_funds_to_buy),
            plural_assets_out_of_funds,
            np.round(root.config.budget, 2),
            root.config.amount_invest_per_asset,
            assets_out_of_funds_to_buy,
        )
    if assets_max_amount_bought:
        root.log.info(
            "%s asset%s %s reached max amount to buy: %s",
            len(assets_max_amount_bought),
            plural_assets_max_amount_bought,
            "have" if len(assets_max_amount_bought) > 1 else "has",
            assets_max_amount_bought,
        )


def _get_buy_volume_asset(root: Tradeforce, asset_symbol: str) -> float:
    """Gets the buy volume of an asset.

    Params:
        root:         Tradeforce main instance.
        asset_symbol: The symbol of the asset.

    Returns:
        The buy volume / balance available of the asset.
    """
    return root.trader.wallets[asset_symbol].balance_available


async def _build_open_order(root: Tradeforce, buy_order: Order) -> dict:
    """Builds an open order dictionary based on the provided buy order.

    Params:
        root:      Tradeforce main instance.
        buy_order: The buy order object.

    Returns:
        The open order dictionary.
    """
    asset_symbol, base_currency = convert_symbol_from_exchange(buy_order.symbol)
    buy_order_fee = np.round(abs(buy_order.fee), 5)
    buy_volume_fiat = np.round(root.config.amount_invest_per_asset - buy_order_fee, 5)
    asset_price_profit = _get_significant_digits(buy_order.price * root.config.profit_factor_target, 5)
    await asyncio_sleep(10)
    buy_volume_asset = _get_buy_volume_asset(root, asset_symbol)

    return {
        "trader_id": root.config.trader_id,
        "buy_order_id": buy_order.id,
        "gid": buy_order.gid,
        "timestamp_buy": int(buy_order.mts_create),
        "asset": asset_symbol,
        "base_currency": base_currency,
        "price_buy": buy_order.price,
        "price_profit": asset_price_profit,
        # TODO: "performance": asset["perf"],
        "amount_invest_per_asset": root.config.amount_invest_per_asset,
        "buy_volume_fiat": buy_volume_fiat,
        "buy_fee_fiat": buy_order_fee,
        "buy_volume_asset": buy_volume_asset,
    }


async def buy_confirmed(root: Tradeforce, buy_order: Order) -> None:
    """Handles a confirmed buy order

    by building an open order, adding it to the trader,
    and submitting a sell order which includes the profit factor.

    Params:
        root:      Tradeforce main instance.
        buy_order: The buy order object.
    """
    open_order = await _build_open_order(root, buy_order)
    root.trader.new_order(open_order, "open_orders")

    if not root.config.is_sim:
        await submit_sell_order(root, open_order)
