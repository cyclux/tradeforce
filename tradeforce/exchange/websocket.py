from __future__ import annotations
from typing import TYPE_CHECKING

from asyncio import sleep as asyncio_sleep
from tqdm.asyncio import tqdm

import numpy as np
import pandas as pd


from tradeforce.utils import convert_symbol_from_exchange, convert_symbol_to_exchange
from tradeforce.trader.buys import buy_confirmed
from tradeforce.trader.sells import sell_confirmed
from tradeforce.custom_types import DictCandle

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import Tradeforce
    from logging import Logger
    from bfxapi.models.order import Order  # type: ignore
    from bfxapi.models.wallet import Wallet  # type: ignore
    from bfxapi.models.subscription import Subscription  # type: ignore


def _check_timestamp_difference(log: Logger, start: int, end: int, freq: str) -> np.ndarray:
    """
    Check the difference between the DB and the latest websocket timestamps of candles.
    If there is a discrepancy, the missing candles are requested from the exchange.

    Args:
        log:    Logger object to log debug information.
        start:  Starting timestamp in milliseconds. This is the timestamp of the last candle in the DB.
        end:    Ending timestamp in milliseconds. This is the timestamp of the latest candle in the WS.
        freq:   Frequency of candles.

    Returns:
        A range of timestamps as a NumPy array.
    """

    log.debug("Check delta between DB and WS timestamp of candles: %s (DB) %s (WS)", start, end)
    diff_range = (
        pd.date_range(
            start=pd.to_datetime(start, unit="ms", utc=True),
            end=pd.to_datetime(end, unit="ms", utc=True),
            freq=freq,
            inclusive="neither",
        ).asi8
        // 10**6
    )
    return diff_range


def _convert_order_to_dict(order_obj: Order) -> dict[str, int | float | str]:
    """
    Convert an Order object (bfx specific) into a dictionary.

    Args:
        order_obj: Order object from bitfinex to convert.

    Returns:
        The order as a dictionary.
    """
    order_dict = {
        "symbol": order_obj.symbol,
        "gid": order_obj.gid,
        "mts_update": order_obj.mts_update,
        "price_avg": order_obj.price_avg,
        "amount_orig": order_obj.amount_orig,
    }
    return order_dict


def _get_order_type(ws_order: Order):
    """
    Determine the order type based on the amount.

    Args:
        ws_order: The websocket Order object.

    Returns:
        A string representing the order type, either "buy" or "sell".
    """
    return "buy" if ws_order.amount_orig > 0 else "sell"


def _is_order_closed_and_filled(ws_order_closed: Order):
    """
    Check if the order is closed and fully filled.
    amount_orig and amount_filled are negative for sell orders.
    Take absolute values to ensure correct calculation of the difference.
    0.0000001 is used as a threshold to account for rounding errors.

    Args:
        ws_order_closed: The websocket Order object.

    Returns:
        True if the order is closed and fully filled, otherwise False.
    """
    return abs(abs(ws_order_closed.amount_orig) - abs(ws_order_closed.amount_filled)) < 0.0000001


def _convert_min_to_m(interval: str) -> str:
    """
    Convert the interval from "min" to "m" to match the Bitfinex API.
    """
    return interval.replace("min", "m")


class ExchangeWebsocket:
    """
    Manages the websocket connection to the Bitfinex exchange.

    This class handles subscriptions to candle, order, and wallet updates, processes
    new candle data, and stores the data in the DB. It ensures that the market
    history remains up-to-date and provides real-time information for the trading engine.
    """

    def __init__(self, root: Tradeforce):
        """
        Initialize the ExchangeWebsocket object.

        Args:
            root: The main Tradeforce instance provides access to the API, config and logging.
        """
        self.root = root
        self.config = root.config
        self.log = root.logging.get_logger(__name__)
        self.api_priv = root.exchange_api.api_priv
        self.api_pub = root.exchange_api.api_pub

        self.ws_candle_cache: dict[int, dict[str, float]] = {}
        self.asset_candle_subs: dict[str, Subscription] = {}
        self.prevent_race_condition_cache: list[int] = []
        self.candle_cache_cap = 20
        self.latest_candle_timestamp = 0
        self.current_candle_timestamp = 0
        self.last_candle_timestamp = 0
        self.ws_subs_finished = False
        self.is_set_last_candle_timestamp = False
        self.history_sync_patch_running = False
        self.diff_range_candle_timestamps: np.ndarray | None = None

    ###################
    # Init websockets #
    ###################

    async def ws_run(self):
        """
        Subscribe to public events and run the public websocket.
        """
        self.api_pub.ws.on("connected", self.ws_init_connection)
        self.api_pub.ws.on("new_candle", self.ws_new_candle)
        self.api_pub.ws.on("error", self.ws_error)
        self.api_pub.ws.on("subscribed", self.ws_is_subscribed)
        self.api_pub.ws.on("unsubscribed", self.ws_unsubscribed)
        self.api_pub.ws.on("status_update", self.ws_status)
        self.api_pub.ws.run()

    async def ws_priv_run(self):
        """
        Subscribe to private events and run the private websocket.
        """
        if self.api_priv is not None:
            self.api_priv.ws.on("connected", self.ws_priv_init_connection)
            self.api_priv.ws.on("wallet_snapshot", self.ws_priv_wallet_snapshot)
            self.api_priv.ws.on("wallet_update", self.ws_priv_wallet_update)
            self.api_priv.ws.on("order_confirmed", self.ws_priv_order_confirmed)
            self.api_priv.ws.on("order_closed", self.ws_priv_order_closed)
            self.api_priv.ws.run()

    ######################
    # Round robin caches #
    ######################

    def prune_race_condition_prevention_cache(self):
        """
        Remove the oldest entry from the race condition prevention cache
        if its size exceeds the limit.
        """
        if len(self.prevent_race_condition_cache) > 3:
            del self.prevent_race_condition_cache[0]

    def prune_candle_cache(self):
        """
        Remove the oldest entries from the candle cache if its size exceeds the specified capacity.
        """
        candles_timestamps = self.ws_candle_cache.keys()
        candle_cache_size = len(self.ws_candle_cache.keys())
        while candle_cache_size > self.candle_cache_cap:
            self.log.debug(
                "Deleting %s from candle cache (max cap: %s candles)",
                min(candles_timestamps),
                self.candle_cache_cap,
            )
            # delete oldest candle cache entry
            del self.ws_candle_cache[min(candles_timestamps)]
            candle_cache_size -= 1

    #############################
    # Public websocket channels #
    #############################

    def ws_error(self, ws_error: dict):
        """
        Log websocket errors.

        Args:
            ws_error: The websocket error.
        """
        self.log.error("ws_error: %s", str(ws_error))

    def ws_is_subscribed(self, ws_subscribed: Subscription):
        """
        Update the asset_candle_subs dictionary with the subscribed asset.

        Args:
            ws_subscribed: The Bitfinex specific Subscription object.
        """
        symbol = convert_symbol_from_exchange(ws_subscribed.symbol)[0]
        self.asset_candle_subs[symbol] = ws_subscribed

    def ws_unsubscribed(self, ws_unsubscribed: Subscription):
        """
        Log unsubscribed events.

        Args:
            ws_unsubscribed: The unsubscribed event.
        """
        print("[DEBUG] type ws_unsubscribed", type(ws_unsubscribed))
        self.log.info("Unsubscribed: %s", ws_unsubscribed)

    def ws_status(self, ws_status: dict):
        """
        Log websocket status updates.
        Includes information about the exchange status, maintenance, etc.

        Args:
            ws_status: Emmits the websocket status.
        """
        print("[DEBUG] type ws_status", type(ws_status))
        # TODO: Handle if exchange goes down / maintanance
        # Check availability of exchange periodically ?!
        self.log.warning("Exchange status: %s", ws_status)

    async def ws_subscribe_candles(self, asset_list: list[str]):
        """
        Subscribe to candle data for the specified assets.

        Args:
            asset_list: A list of asset symbols to subscribe to.
        """
        asset_list_bfx = convert_symbol_to_exchange(asset_list)
        async for symbol in tqdm(asset_list_bfx, desc="Subscribing to candles"):
            await self.api_pub.ws.subscribe_candles(symbol, _convert_min_to_m(self.config.candle_interval))
            # Rate limit to avoid hitting the websocket rate limit
            await asyncio_sleep(0.1)
        self.ws_subs_finished = True
        self.log.info("Subscribed to %s channels.", len(asset_list_bfx))

    async def get_latest_candle_timestamp(self) -> int:
        """
        Obtain the latest candle timestamp from either local market history or the remote API.

        This function first checks the local market history for the latest candle timestamp. If it's not available
        (i.e., the value is 0), the function fetches the timestamp from the remote API by using the configured
        history timeframe.

        The fetched timestamp can be used to synchronize data between the local market history and the remote API,
        ensuring that the latest market data is up-to-date for further processing.

        Returns:
            The latest candle timestamp.
        """
        latest_candle_timestamp = self.root.market_history.get_local_candle_timestamp(position="latest")
        if latest_candle_timestamp == 0:
            latest_candle_timestamp = await self.root.exchange_api.get_latest_remote_candle_timestamp(
                minus_delta=f"{self.config.history_timeframe_days}days"
            )
        return latest_candle_timestamp

    async def ws_init_connection(self):
        """
        Initialize the public websocket connection, fetch the latest candle timestamp, and subscribe to candle updates.

        Obtains the latest candle timestamp from either local market history or the remote API by
        calling the 'get_latest_candle_timestamp()' method. After setting the initial timestamp subscribe to
        candle updates for the specified asset list.

        The 'is_set_last_candle_timestamp' flag is used to indicate whether the initial candle timestamp has been set.
        This is used to prevent the 'process_candle_history_sync()' method
        from being called before the initial candle is received.
        """
        self.latest_candle_timestamp = await self.get_latest_candle_timestamp()
        self.is_set_last_candle_timestamp = False

        self.log.info("Subscribing to %s websocket channels..", self.config.exchange)
        await self.ws_subscribe_candles(self.root.assets_list_symbols)

    async def ws_priv_init_connection(self):
        """
        Log the initialization of the Live Trading API.
        """
        self.log.info("Live Trading API initialized. Connected to private websocket channels.")

    async def ws_new_candle(self, candle: DictCandle):
        """
        Process and handle new candle updates from the websocket.

        Args:
            candle: A custom dictionary containing the new candle data.
        """
        self.current_candle_timestamp = int(candle["mts"])
        self.update_candle_cache(candle)
        await self.process_candle_history_sync()
        await self.handle_new_candle()

    def update_candle_cache(self, candle: DictCandle):
        """
        Update the websocket candle cache with the new candle data.

        Args:
            candle: A custom dictionary containing the new candle data.
        """
        symbol_converted = convert_symbol_from_exchange(candle["symbol"])[0]
        current_asset = self.ws_candle_cache.get(self.current_candle_timestamp, {})
        current_asset[f"{symbol_converted}_o"] = candle["open"]
        current_asset[f"{symbol_converted}_c"] = candle["close"]
        current_asset[f"{symbol_converted}_h"] = candle["high"]
        current_asset[f"{symbol_converted}_l"] = candle["low"]
        current_asset[f"{symbol_converted}_v"] = candle["volume"]
        self.ws_candle_cache[self.current_candle_timestamp] = current_asset

    async def process_candle_history_sync(self):
        """
        Process the synchronization of candle history.
        Set the last candle timestamp and check if history patching is needed.
        If so, patch the history accordingly.
        """
        if not self.is_set_last_candle_timestamp:
            self.set_last_candle_timestamp()
            if self.should_patch_history():
                await self.patch_history()

    def set_last_candle_timestamp(self):
        """
        Set the last_candle_timestamp attribute if there are at least 2 candles in the cache.
        """
        candles_timestamps = self.ws_candle_cache.keys()
        candle_cache_size = len(candles_timestamps)
        if candle_cache_size >= 2:
            self.is_set_last_candle_timestamp = True
            self.last_candle_timestamp = max(candles_timestamps)
            self.log.debug("last_candle_timestamp set to %s", self.last_candle_timestamp)

    def should_patch_history(self):
        """
        Determine if the history should be patched based on the difference in candle timestamps.

        Returns:
            True if the history should be patched, otherwise False.
        """
        diff_range_candle_timestamps = _check_timestamp_difference(
            self.log,
            start=self.latest_candle_timestamp,
            end=self.last_candle_timestamp,
            freq=self.config.candle_interval,
        )
        # If there are differences in the candle timestamps, patch the history
        # from the earliest (min) to the latest (max) timestamp
        if len(diff_range_candle_timestamps) > 1:
            self.timestamp_patch_history_start = min(diff_range_candle_timestamps)
            self.timestamp_patch_history_end = max(diff_range_candle_timestamps)
            return self.timestamp_patch_history_start != self.timestamp_patch_history_end
        return False

    async def patch_history(self):
        """
        Patch the out-of-sync history from the specified start and end timestamps.
        Update the market history accordingly.
        The flag history_sync_patch_running is set to block potentially conflicting updates.
        """
        self.history_sync_patch_running = True
        self.log.info(
            "Patching out of sync history.. From %s to %s",
            self.timestamp_patch_history_start,
            self.timestamp_patch_history_end,
        )
        await self.root.market_history.update(
            start=self.timestamp_patch_history_start, end=self.timestamp_patch_history_end
        )
        self.history_sync_patch_running = False

    async def handle_new_candle(self):
        """
        Handle new candle data received from the websocket.
        Process the new candle if conditions are met -> see is_new_candle()
        and trigger trader updates if configured to run live.
        """
        if self.is_new_candle():
            self.prevent_race_condition_cache.append(self.current_candle_timestamp)
            await self.save_new_candle_to_db()
            if self.config.run_live:
                await self.trigger_trader_updates()

    def is_new_candle(self):
        """
        Check if the current candle is new and if conditions are met to process it.

        Returns:
            True under following conditions:
            1. The current candle timestamp is not in the race condition prevention cache,
            ensuring that the same candle is not processed multiple times due to overlapping updates.
            2. The current candle timestamp is greater than the last processed candle timestamp,
            ensuring that the new candle is indeed more recent than the previous one.
            3. The last candle timestamp flag (is_set_last_candle_timestamp) is set,
            indicating that the last candle timestamp has been successfully retrieved.
            4. The websocket subscription process (ws_subs_finished) has been completed,
            ensuring that the websocket is ready to receive and process candle updates.
        """
        return (
            self.current_candle_timestamp not in self.prevent_race_condition_cache
            and self.current_candle_timestamp > self.last_candle_timestamp
            and self.is_set_last_candle_timestamp
            and self.ws_subs_finished
        )

    async def save_new_candle_to_db(self):
        """
        Save the new candle to the database and update the last_candle_timestamp.
        Prune the candle cache and race condition prevention cache.
        """
        self.log.info("New candle received [timestamp: %s] - Saved to %s", self.last_candle_timestamp, self.config.dbms)
        candles_last_timestamp = self.ws_candle_cache.get(self.last_candle_timestamp, {})
        df_history_update = pd.DataFrame(candles_last_timestamp, index=[self.last_candle_timestamp])
        self.last_candle_timestamp = self.current_candle_timestamp
        df_history_update.index.name = "t"
        self.root.backend.db_add_history(df_history_update)
        self.prune_candle_cache()
        self.prune_race_condition_prevention_cache()

    async def trigger_trader_updates(self):
        """
        Trigger trader updates based on the current configuration.
        Check for sold orders and update the trader if not running in simulation mode.
        Log the current total profit.
        """
        if self.root.exchange_api.api_priv is not None:
            await self.root.trader.check_sold_orders()
        if not self.history_sync_patch_running and not self.config.is_sim:
            await self.root.trader.candles_update()
            current_total_profit = self.root.trader.get_profit()
            self.log.info("Current total profit: $%s", current_total_profit)

    ##############################
    # Private websocket channels #
    ##############################

    # Handle order confirmed

    def ws_priv_order_confirmed(self, ws_confirmed: Order):
        """
        Handle the confirmation of an order from the private websocket.

        Args:
            ws_confirmed: A Bitfinex specific Order object.
        """
        self.log.debug("order_confirmed: %s", str(ws_confirmed))
        asset_symbol = convert_symbol_from_exchange(ws_confirmed.symbol)[0]

        buy_order = {"asset": asset_symbol, "gid": ws_confirmed.gid}
        if _get_order_type(ws_confirmed) == "sell":
            self.handle_sell_order(buy_order, ws_confirmed)
        else:
            self.log.error("Cannot find open order (%s)", str(buy_order))

    def handle_sell_order(self, buy_order: dict, ws_confirmed: Order):
        """
        Handle a sell order by updating the open order.

        Args:
            buy_order:    A dictionary containing the buy order information.
            ws_confirmed: A Bitfinex specific Order object.
        """
        open_order = self.root.trader.get_open_order(asset=buy_order)
        if len(open_order) > 0:
            self.update_open_order(open_order[0], ws_confirmed)
        else:
            self.log.error("Cannot find open order (%s)", str(buy_order))

    def update_open_order(self, open_order: dict, ws_confirmed: Order):
        """
        Update an open order with the sell order ID from the websocket data.

        Args:
            open_order:   A dictionary containing the open order's details.
            ws_confirmed: A Bitfinex specific Order object containing the sell order ID.
        """
        open_order["sell_order_id"] = ws_confirmed.id
        self.root.trader.edit_order(open_order, "open_orders")

    # Handle order closed

    async def ws_priv_order_closed(self, ws_order_closed: Order):
        """
        Handle a websocket event for a closed order.

        Args:
            ws_order_closed: A Bitfinex specific Order object.
        """
        self.log.debug("order_closed: %s", str(ws_order_closed))
        if _is_order_closed_and_filled(ws_order_closed):
            await self.handle_order_closed(ws_order_closed)

    async def handle_order_closed(self, ws_order_closed: Order):
        """
        Process the closed order based on its type (buy or sell).

        Args:
            ws_order_closed: A Bitfinex specific Order object.
        """
        order_type = _get_order_type(ws_order_closed)
        if order_type == "buy":
            await buy_confirmed(self.root, ws_order_closed)
        if order_type == "sell":
            order_closed_dict = _convert_order_to_dict(ws_order_closed)
            sell_confirmed(self.root, order_closed_dict)

    # Handle wallet updates

    async def ws_priv_wallet_snapshot(self, ws_wallet_snapshot: list[Wallet]):
        """
        Utilize the wallet snapshot to set the trader's budget and update balances.
        Also trigger the minimum order size update.

        Args:
            ws_wallet_snapshot: List of Wallets containing the wallet balances.
        """
        self.log.debug("wallet_snapshot: %s", str(ws_wallet_snapshot))
        self.root.trader.set_budget(ws_wallet_snapshot)
        self.root.backend.db_sync_state_trader()
        self.root.backend.db_sync_state_orders()
        await self.root.trader.get_min_order_sizes()

    def ws_priv_wallet_update(self, ws_wallet_update: Wallet):
        """
        Update the Wallets with the latest balances from the websocket event.
        The balance of the base currency is equivalent to the trader's budget. e.g. USD

        Args:
            ws_wallet_update: A Wallet instance containing the updated wallet information.
        """
        if ws_wallet_update.currency == self.config.base_currency:
            self.root.trader.set_budget([ws_wallet_update])
        else:
            self.root.trader.wallets[ws_wallet_update.currency] = ws_wallet_update

    # def check_ws_health(self, health_check_size=10):
    #     health_check_size *= -1
    #     latest_candle_timestamps = sorted(self.ws_candle_cache.keys())[health_check_size:]
    #     healthy_assets = {}

    #     for candle_timestamp in latest_candle_timestamps:
    #         assets = get_col_names(self.ws_candle_cache[candle_timestamp])
    #         # collect how often an asset candle appears within health_check_size
    #         for asset in assets:
    #             current_asset = healthy_assets.get(asset, 0)
    #             healthy_assets[asset] = current_asset + 1
    #     # for now we only check if a candle of an asset was received within health_check_size
    #     # if no candle was received within health_check_size the asset is considered "unhealthty"
    #     # if at least one candle was received, the asset is considered "healthy"
    #     healthy_assets_list = list(healthy_assets.keys())
    #     unhealthy_assets = np.setdiff1d(self.root.assets_list_symbols, healthy_assets_list)
    #     not_subscribed_anymore = []

    #     # check if still subscribed
    #     for asset in unhealthy_assets:
    #         asset_is_subbed = False
    #         asset_sub = self.asset_candle_subs.get(asset, None)
    #         if asset_sub is not None:
    #             asset_is_subbed = asset_sub.is_subscribed()
    #         if not asset_is_subbed:
    #             not_subscribed_anymore.append(asset)

    #     return {"healthy": healthy_assets_list,
    #            "unhealthy": unhealthy_assets,
    #            "not_subscribed": not_subscribed_anymore}
