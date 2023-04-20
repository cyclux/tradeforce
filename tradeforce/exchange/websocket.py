""" exchange/websocket.py

Module: WebSocket Module
------------------------

This module provides functionality for handling real-time data updates from the Bitfinex exchange
through websocket connections. The module processes new candle updates and maintains a cache of
recent candles to ensure continuity in the data. Additionally handles synchronization of candle
history and patches any missing data.

The module also manages private websocket channels for live trading:
order confirmations, closed orders, and wallet updates. Order confirmations are handled by
updating open orders with sell order IDs, while closed orders are processed based on their
type (buy or sell). Wallet updates include updating the trader's budget and wallet balances.

Main ExchangeWebsocket methods:

    ws_run:      Responsible for running the public websocket and subscribing to public events such
                    as candle updates. It sets up the event listeners for various events including
                    initialization, new candles, errors, subscription status, and websocket status
                    updates. Once configured, it starts the public websocket.

    ws_priv_run: This method handles the private websocket, which deals with user-specific data
                    such as wallet updates and order confirmations. It sets up event listeners for
                    private events like wallet snapshots, wallet updates, order confirmations, and
                    order closures. The private websocket is run only if the private API is available.

"""


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
    """Check the difference between the DB and the latest websocket timestamps of candles.

    Params:
        log:    Logger object to log debug information.
        start:  Starting timestamp in ms. This is the _init_reference_timestamp from DB or API.
        end:    Ending timestamp in ms. This is the last_candle_timestamp from the websocket.
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
    """Convert an bfx websocket Order object into a dictionary.

    Params:
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


def _get_order_type(ws_order: Order) -> str:
    """Determine the order type based on the amount

    within the bfx websocket Order object:
    Positive amounts are buy orders, negative amounts are sell orders.

    Params:
        ws_order: The websocket Order object.

    Returns:
        A string representing the order type, either "buy" or "sell".
    """
    return "buy" if ws_order.amount_orig > 0 else "sell"


def _is_order_closed_and_filled(ws_order_closed: Order) -> bool:
    """Check if the order is closed and fully filled.

    amount_orig and amount_filled are negative for sell orders.
    Thus, take absolute values to ensure correct calculation of the difference.
    Threshold accounts for rounding errors.

    Params:
        ws_order_closed: The websocket Order object.

    Returns:
        True if the order is closed and fully filled, otherwise False.
    """
    threshold = 0.0000001
    return abs(abs(ws_order_closed.amount_orig) - abs(ws_order_closed.amount_filled)) < threshold


def _convert_min_to_m(interval: str) -> str:
    """Convert the interval from "min" to "m" to match the Bitfinex API.

    Params:
        interval: The interval to convert.

    Returns:
        The converted interval.
    """
    return interval.replace("min", "m")


class ExchangeWebsocket:
    """Manages the websocket connection to the Bitfinex exchange.

    Handles subscriptions to candle, order, and wallet updates, processes
    new candle data, and stores the data in the DB. It ensures that the
    market history remains up-to-date and provides real-time information
    for the trading engine.
    """

    def __init__(self, root: Tradeforce):
        """Initialize the ExchangeWebsocket instance.

        Params:
            root: The main Tradeforce instance provides access to the API,
            config and logging or any other module.
        """
        self.root = root
        self.config = root.config
        self.log = root.logging.get_logger(__name__)

        if hasattr(root.exchange_api, "api_priv"):
            self.api_priv = root.exchange_api.api_priv
        if hasattr(root.exchange_api, "api_pub"):
            self.api_pub = root.exchange_api.api_pub

        self.ws_candle_cache: dict[int, dict[str, float]] = {}
        self.asset_candle_subs: dict[str, Subscription] = {}
        self.prevent_race_condition_cache: list[int] = []
        self.candle_cache_cap = 20
        self._init_reference_timestamp = 0
        self.current_candle_timestamp = 0
        self.last_candle_timestamp = 0
        self.ws_subs_finished = False
        self.is_set_init_reference_timestamp = False
        self.history_sync_patch_running = False
        self.diff_range_candle_timestamps: np.ndarray | None = None

    # ----------------
    # Init websockets
    # ----------------

    async def ws_run(self) -> None:
        """Run the public websocket and subscribe to public events."""
        if hasattr(self, "api_pub"):
            self.api_pub.ws.on("connected", self._ws_init_connection)
            self.api_pub.ws.on("new_candle", self._ws_new_candle)
            self.api_pub.ws.on("error", self._ws_error)
            self.api_pub.ws.on("subscribed", self._ws_is_subscribed)
            self.api_pub.ws.on("unsubscribed", self._ws_unsubscribed)
            self.api_pub.ws.on("status_update", self._ws_status)
            self.api_pub.ws.run()

    async def ws_priv_run(self) -> None:
        """Run the private websocket and subscribe to private events."""

        if hasattr(self, "api_priv"):
            self.api_priv.ws.on("connected", self._ws_priv_init_connection)
            self.api_priv.ws.on("wallet_snapshot", self._ws_priv_wallet_snapshot)
            self.api_priv.ws.on("wallet_update", self._ws_priv_wallet_update)
            self.api_priv.ws.on("order_confirmed", self._ws_priv_order_confirmed)
            self.api_priv.ws.on("order_closed", self._ws_priv_order_closed)
            self.api_priv.ws.run()

    # -------------------
    # Round robin caches
    # -------------------

    def _prune_race_condition_prevention_cache(self) -> None:
        """Remove the oldest entry from the race condition prevention cache
        -> if its size exceeds the limit (3).
        """
        if len(self.prevent_race_condition_cache) > 3:
            del self.prevent_race_condition_cache[0]

    def _prune_candle_cache(self) -> None:
        """Remove the oldest entries from the candle cache
        -> if its size exceeds candle_cache_cap."""

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

    # --------------------------
    # Public websocket channels
    # --------------------------

    def _ws_error(self, ws_error: dict) -> None:
        """Log websocket errors.

        Params:
            ws_error: The websocket error.
        """
        self.log.error("ws_error: %s", str(ws_error))

    def _ws_is_subscribed(self, ws_subscribed: Subscription) -> None:
        """Update the asset_candle_subs dictionary
            with the subscribed asset.

        Params:
            ws_subscribed: The Bitfinex specific Subscription object.
        """
        symbol = convert_symbol_from_exchange(ws_subscribed.symbol)[0]
        self.asset_candle_subs[symbol] = ws_subscribed

    def _ws_unsubscribed(self, ws_unsubscribed: Subscription) -> None:
        """Log unsubscribed events.

        Params:
            ws_unsubscribed: The unsubscribed event.
        """
        print("[DEBUG] type ws_unsubscribed", type(ws_unsubscribed))
        self.log.info("Unsubscribed: %s", ws_unsubscribed)

    def _ws_status(self, ws_status: dict) -> None:
        """Log websocket status updates.
            Includes info about the exchange status, maintenance, etc.

        Params:
            ws_status: Emmits the websocket status.
        """
        print("[DEBUG] type ws_status", type(ws_status))

        # TODO: Handle if exchange goes down / maintanance
        self.log.warning("Exchange status: %s", ws_status)

    async def _ws_subscribe_candles(self, asset_list: list[str]) -> None:
        """Subscribe to candle data for the specified assets.

        tqdm is utilized to display a progress bar.
        Sleep 0.1 seconds between each subscription
        to avoid hitting the websocket rate limit.

        Params:
            asset_list: A list of asset symbols to subscribe to.
        """
        if len(asset_list) == 0:
            self.log.warning("No assets to subscribe to.")
            return

        asset_list_bfx = convert_symbol_to_exchange(asset_list)
        async for symbol in tqdm(asset_list_bfx, desc="Subscribing to candles"):
            await self.api_pub.ws.subscribe_candles(symbol, _convert_min_to_m(self.config.candle_interval))
            # Rate limit to avoid hitting the websocket rate limit
            await asyncio_sleep(0.1)

        self.ws_subs_finished = True

        self.log.info("Subscribed to %s channels.", len(asset_list_bfx))

    async def _get_init_reference_timestamp(self) -> int:
        """Obtain the initial reference candle timestamp

        from either local market history or the remote API.

        First check the local market history for the latest candle timestamp.
        In case it is present, update the missing candles until now from the
        remote API.

        If it's not present, fetch the timestamp from the remote API by using
        the configured history timeframe to find a starting point and
        thereby establishing the initial candle dataset.

        Returns:
            The initial reference candle timestamp.
        """
        init_reference_timestamp = self.root.market_history.get_local_candle_timestamp(position="latest")

        if init_reference_timestamp == 0:
            init_reference_timestamp = await self.root.exchange_api.get_latest_remote_candle_timestamp(
                minus_delta=f"{self.config.fetch_init_timeframe_days}days"
            )

        return init_reference_timestamp

    async def _ws_init_connection(self) -> None:
        """Initialize the public websocket connection
        -> Fetch the 'init_reference_timestamp' and subscribe to candle updates.

        Wait for the market history to finish loading:
        This ensures that the local market history and 'asset_list_symbols'
        is up-to-date.

        The 'is_set_init_reference_timestamp' flag is used to indicate whether
        the initial candle timestamp has been set. This is used to prevent the
        'process_candle_history_sync()' method from being called before the
        initial candle is received.
        """

        await self.root.market_history.is_load_history_finished.wait()

        self._init_reference_timestamp = await self._get_init_reference_timestamp()
        self.is_set_init_reference_timestamp = False

        self.log.info("Subscribing to %s websocket channels..", self.config.exchange)
        await self._ws_subscribe_candles(self.root.asset_symbols)

    async def _ws_priv_init_connection(self) -> None:
        """Log the initialization of the Live Trading API."""
        self.log.info("Live Trading API initialized. Connected to private websocket channels.")

    async def _ws_new_candle(self, candle: DictCandle) -> None:
        """Process and handle new candle updates from the websocket.

        DictCandle = TypedDict(
            "DictCandle",
            {
                "mts": int,
                "open": float,
                "close": float,
                "high": float,
                "low": float,
                "volume": float,
                "symbol": str,
            },
        )

        Params:
            candle: A custom dictionary containing the new candle data.
        """
        # 'mts' is the candle timestamp in milliseconds from bfx exchange.
        self.current_candle_timestamp = int(candle["mts"])
        self._update_candle_cache(candle)

        if not self.is_set_init_reference_timestamp:
            await self._process_candle_history_sync()

        await self._handle_new_candle()

    def _update_candle_cache(self, candle: DictCandle) -> None:
        """Update the websocket candle cache
        -> with the new candle data.

        Params:
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

    async def _process_candle_history_sync(self) -> None:
        """Process the synchronization / patching of candle history."""

        self._set_last_candle_timestamp()

        if self._should_patch_history():
            await self._patch_history()

    def _set_last_candle_timestamp(self) -> None:
        """Set the 'last_candle_timestamp' attribute
        -> if there are at least 2 candles in the cache.
        """
        candles_timestamps = self.ws_candle_cache.keys()
        candle_cache_size = len(candles_timestamps)

        if candle_cache_size >= 2:
            self.is_set_init_reference_timestamp = True
            self.last_candle_timestamp = max(candles_timestamps)
            self.log.debug("'last_candle_timestamp' set to %s", self.last_candle_timestamp)

    def _should_patch_history(self) -> bool:
        """Determine if the history should be patched (requested from the exchange)

        based on the difference in candle timestamps. Currently True if there are
        3 or more timestamps missing between the '_init_reference_timestamp' and
        'last_candle_timestamp'.

        Returns:
            True if the history should be patched, otherwise False.
        """
        self.diff_range_candle_timestamps = _check_timestamp_difference(
            self.log,
            start=self._init_reference_timestamp,
            end=self.last_candle_timestamp,
            freq=self.config.candle_interval,
        )

        if self.diff_range_candle_timestamps is None:
            return False

        # If there are differences in the candle timestamps, patch the history
        # from the earliest (min) to the latest (max) timestamp
        if len(self.diff_range_candle_timestamps) >= 3:
            self.timestamp_patch_history_start = min(self.diff_range_candle_timestamps)
            self.timestamp_patch_history_end = max(self.diff_range_candle_timestamps)
            return self.timestamp_patch_history_start != self.timestamp_patch_history_end

        return False

    async def _patch_history(self) -> None:
        """Patch out-of-sync history

        from the specified start and end timestamps. Update the market history accordingly.
        The flag 'history_sync_patch_running' is set to block potentially conflicting updates.
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

    async def _handle_new_candle(self) -> None:
        """Handle new candle data received from the websocket.

        Process the new candle if conditions are met -> see is_new_candle()
        and trigger trader updates if configured to run live trading.

        'prevent_race_condition_cache' is used to prevent the same candle from being processed
        multiple times as the asynchronous websocket updates may overlap.
        """
        if self._is_new_candle():
            self.prevent_race_condition_cache.append(self.current_candle_timestamp)
            await self._save_new_candle_to_db()
            if self.config.run_live:
                await self._trigger_trader_updates()

    def _is_new_candle(self) -> bool:
        """Check if the current candle is new
        -> and thus conditions are met to process it.

        Returns True under following conditions:

            1. The current candle timestamp is not in the race condition prevention
            cache, ensuring that the same candle is not processed multiple times due
            to overlapping updates.

            2. The current candle timestamp is greater than the last processed candle
            timestamp, ensuring that the new candle is indeed more recent than the
            previous one.

            3. The last candle timestamp flag (is_set_init_reference_timestamp) is
            set, indicating that the last candle timestamp has been successfully
            retrieved.

            4. The websocket subscription process (ws_subs_finished) has been completed,
            ensuring that the websocket is ready to receive and process candle updates.
        """
        return (
            self.current_candle_timestamp not in self.prevent_race_condition_cache
            and self.current_candle_timestamp > self.last_candle_timestamp
            and self.is_set_init_reference_timestamp
            and self.ws_subs_finished
        )

    def _prepare_history_update(self) -> pd.DataFrame:
        """Prepare the history update DataFrame

        by retrieving the last candle from the candle cache,
        converting it to a DataFrame and setting the index.

        Returns:
            The history update DataFrame
        """
        candles_last_timestamp = self.ws_candle_cache.get(self.last_candle_timestamp, {})
        df_history_update = pd.DataFrame(candles_last_timestamp, index=[self.last_candle_timestamp])
        df_history_update.index.name = "t"

        return df_history_update

    async def _save_new_candle_to_db(self) -> None:
        """Save the new candle to the database

        and update the last_candle_timestamp. Prune the candle cache
        and race condition prevention cache.
        """
        self.log.info("New candle received [timestamp: %s] - Saved to %s", self.last_candle_timestamp, self.config.dbms)

        df_history_update = self._prepare_history_update()

        self.last_candle_timestamp = self.current_candle_timestamp

        self.root.backend.db_add_history(df_history_update)

        self._prune_candle_cache()
        self._prune_race_condition_prevention_cache()

    async def _trigger_trader_updates(self) -> None:
        """Trigger trader updates based on the current configuration.

        Check for sold orders and update the trader if not running in simulation mode.
        Log the current total profit.
        """
        if self.root.exchange_api.api_priv is not None:
            await self.root.trader.check_sold_orders()

        if not self.history_sync_patch_running and not self.config.is_sim:
            await self.root.trader.candles_update()
            current_total_profit = self.root.trader.get_profit()
            self.log.info("Current total profit: $%s", current_total_profit)

    # ---------------------------
    # Private websocket channels
    # ---------------------------

    # Handle order confirmed

    def _ws_priv_order_confirmed(self, ws_confirmed: Order) -> None:
        """Handle the confirmation of an order from the private websocket.

        Params:
            ws_confirmed: A Bitfinex specific Order object.
        """
        self.log.debug("order_confirmed: %s", str(ws_confirmed))

        # Convert the asset symbol e.g. tBTCUSD -> BTC
        asset_symbol = convert_symbol_from_exchange(ws_confirmed.symbol)[0]

        # Construct a buy order dict to find the open order
        buy_order = {"asset": asset_symbol, "gid": ws_confirmed.gid}

        if _get_order_type(ws_confirmed) == "sell":
            self._handle_sell_order(buy_order, ws_confirmed)
        else:
            self.log.error("Cannot find open order (%s)", str(buy_order))

    def _handle_sell_order(self, buy_order: dict, ws_confirmed: Order) -> None:
        """Handle a sell order
        -> by updating the open order.

        Params:
            buy_order:    Dict containing the buy order information.
            ws_confirmed: A Bitfinex specific Order object.
        """
        open_order = self.root.trader.get_open_order(asset=buy_order)

        if len(open_order) > 0:
            self._update_open_order(open_order[0], ws_confirmed)
        else:
            self.log.error("Cannot find open order (%s)", str(buy_order))

    def _update_open_order(self, open_order: dict, ws_confirmed: Order) -> None:
        """Update an open order
        -> with the sell order ID from the websocket data.

        Params:
            open_order:   Dict containing the open order's details.
            ws_confirmed: A Bitfinex specific Order object containing the sell order ID.
        """
        open_order["sell_order_id"] = ws_confirmed.id
        self.root.trader.edit_order(open_order, "open_orders")

    # Handle order closed

    async def _ws_priv_order_closed(self, ws_order_closed: Order) -> None:
        """Handle a websocket event
        -> for a closed order.

        Params:
            ws_order_closed: A Bitfinex specific Order object.
        """
        self.log.debug("order_closed: %s", str(ws_order_closed))

        if _is_order_closed_and_filled(ws_order_closed):
            await self._handle_order_closed(ws_order_closed)

    async def _handle_order_closed(self, ws_order_closed: Order) -> None:
        """Handle the closed order
                based on its type (buy or sell).

        Params:
            ws_order_closed: A Bitfinex specific Order object.
        """
        order_type = _get_order_type(ws_order_closed)

        if order_type == "buy":
            await buy_confirmed(self.root, ws_order_closed)
        if order_type == "sell":
            order_closed_dict = _convert_order_to_dict(ws_order_closed)
            sell_confirmed(self.root, order_closed_dict)

    # Handle wallet updates

    async def _ws_priv_wallet_snapshot(self, ws_wallet_snapshot: list[Wallet]) -> None:
        """Utilize the wallet snapshot

        to set the trader's budget and update balances,
        also sync states of trader and order
        and trigger the minimum order size update.

        Params:
            ws_wallet_snapshot: List of Wallets containing the wallet balances.
        """
        self.log.debug("wallet_snapshot: %s", str(ws_wallet_snapshot))

        self.root.trader.set_budget(ws_wallet_snapshot)

        self.root.backend.db_sync_state_trader()
        self.root.backend.db_sync_state_orders()

        await self.root.trader.get_min_order_sizes()

    def _ws_priv_wallet_update(self, ws_wallet_update: Wallet) -> None:
        """Update the Wallets

        with the latest balances from the websocket event. The balance
        of the base currency is equivalent to the trader's budget. e.g. USD

        Params:
            ws_wallet_update: A Wallet instance containing the updated wallet information.
        """
        if ws_wallet_update.currency == self.config.base_currency:
            self.root.trader.set_budget([ws_wallet_update])
        else:
            self.root.trader.wallets[ws_wallet_update.currency] = ws_wallet_update

    # def check_ws_health(self, health_check_size=10):
    #     health_check_size *= -1
    #     latest_db_candle_timestamps = sorted(self.ws_candle_cache.keys())[health_check_size:]
    #     healthy_assets = {}

    #     for candle_timestamp in latest_db_candle_timestamps:
    #         assets = get_col_names(self.ws_candle_cache[candle_timestamp])
    #         # collect how often an asset candle appears within health_check_size
    #         for asset in assets:
    #             current_asset = healthy_assets.get(asset, 0)
    #             healthy_assets[asset] = current_asset + 1
    #     # for now we only check if a candle of an asset was received within health_check_size
    #     # if no candle was received within health_check_size the asset is considered "unhealthty"
    #     # if at least one candle was received, the asset is considered "healthy"
    #     healthy_assets_list = list(healthy_assets.keys())
    #     unhealthy_assets = np.setdiff1d(self.root.asset_symbols, healthy_assets_list)
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
