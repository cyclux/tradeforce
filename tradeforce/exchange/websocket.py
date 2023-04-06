"""_summary_

Returns:
    _type_: _description_
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


def check_timestamp_difference(log: Logger, start: int, end: int, freq="5min") -> np.ndarray:
    """Check the difference between the database and websocket timestamps of candles.

    :param log: Logger object
    :param start: Starting timestamp in milliseconds
    :param end: Ending timestamp in milliseconds
    :param freq: Frequency of candles (e.g., '5min')
    :return: A range of timestamps as a NumPy array
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


def convert_order_to_dict(order_obj: Order) -> dict[str, int | float | str]:
    """Convert an order object into a dictionary.

    :param order_obj: Order object from bitfinex to convert
    :return: The order as a dictionary
    """
    order_dict = {
        "symbol": order_obj.symbol,
        "gid": order_obj.gid,
        "mts_update": order_obj.mts_update,
        "price_avg": order_obj.price_avg,
        "amount_orig": order_obj.amount_orig,
    }
    return order_dict


def get_order_type(ws_order):
    return "buy" if ws_order.amount_orig > 0 else "sell"


def is_order_closed_and_filled(ws_order_closed):
    return abs(abs(ws_order_closed.amount_orig) - abs(ws_order_closed.amount_filled)) < 0.0000001


class ExchangeWebsocket:
    """Manages the websocket connection to the Bitfinex exchange.

    This class handles subscriptions to candle, order, and wallet updates, processes
    new candle data, and stores the data in the DB. It ensures that the market
    history remains up-to-date and provides real-time information for the trading engine.
    """

    def __init__(self, root: Tradeforce):
        """Initialize the ExchangeWebsocket object.

        :param root: The Tradeforce object providing access to the config and logging
        """
        self.root = root
        self.config = root.config
        self.log = root.logging.get_logger(__name__)
        self.bfx_api_priv = root.exchange_api.api.get("bfx_api_priv", None)
        self.bfx_api_pub = root.exchange_api.api.get("bfx_api_pub", None)

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
        """Subscribe to public events and run the public websocket."""
        # print("[DEBUG] Starting websocket PUB")
        self.bfx_api_pub.ws.on("connected", self.ws_init_connection)
        self.bfx_api_pub.ws.on("new_candle", self.ws_new_candle)
        self.bfx_api_pub.ws.on("error", self.ws_error)
        self.bfx_api_pub.ws.on("subscribed", self.ws_is_subscribed)
        self.bfx_api_pub.ws.on("unsubscribed", self.ws_unsubscribed)
        self.bfx_api_pub.ws.on("status_update", self.ws_status)
        self.bfx_api_pub.ws.run()

    async def ws_priv_run(self):
        """Subscribe to private events and run the private websocket."""
        if self.bfx_api_priv is not None:
            # print("[DEBUG] Starting websocket PRIV")
            self.bfx_api_priv.ws.on("connected", self.ws_priv_init_connection)
            self.bfx_api_priv.ws.on("wallet_snapshot", self.ws_priv_wallet_snapshot)
            self.bfx_api_priv.ws.on("wallet_update", self.ws_priv_wallet_update)
            self.bfx_api_priv.ws.on("order_confirmed", self.ws_priv_order_confirmed)
            self.bfx_api_priv.ws.on("order_closed", self.ws_priv_order_closed)
            self.bfx_api_priv.ws.run()

    ######################
    # Round robin caches #
    ######################

    def prune_race_condition_prevention_cache(self):
        if len(self.prevent_race_condition_cache) > 3:
            del self.prevent_race_condition_cache[0]

    def prune_candle_cache(self):
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

    def ws_error(self, ws_error):
        self.log.error("ws_error: %s", str(ws_error))

    def ws_is_subscribed(self, ws_subscribed: Subscription):
        symbol = convert_symbol_from_exchange(ws_subscribed.symbol)[0]
        self.asset_candle_subs[symbol] = ws_subscribed

    def ws_unsubscribed(self, ws_unsubscribed):
        self.log.info("Unsubscribed: %s", ws_unsubscribed)

    def ws_status(self, ws_status):
        # TODO: Handle if exchange goes down / maintanance
        # Check availability of exchange periodically ?!
        self.log.warning("Exchange status: %s", ws_status)

    async def ws_subscribe_candles(self, asset_list):
        asset_list_bfx = convert_symbol_to_exchange(asset_list)
        async for symbol in tqdm(asset_list_bfx, desc="Subscribing to candles"):
            # candle_interval[:-2] to convert "5min" -> "5m"
            await self.bfx_api_pub.ws.subscribe_candles(symbol, self.config.candle_interval[:-2])
            await asyncio_sleep(0.1)
        self.ws_subs_finished = True
        self.log.info("Subscribed to %s channels.", len(asset_list_bfx))

    async def get_latest_candle_timestamp(self):
        """Get the latest local candle timestamp, or fetch it from the remote API if not available."""
        latest_candle_timestamp = self.root.market_history.get_local_candle_timestamp(position="latest")
        if latest_candle_timestamp == 0:
            latest_candle_timestamp = await self.root.exchange_api.get_latest_remote_candle_timestamp(
                minus_delta=self.config.history_timeframe
            )
        return latest_candle_timestamp

    async def ws_init_connection(self):
        """Initialize the public websocket connection and subscribe to candle updates."""
        # Get the latest candle timestamp
        self.latest_candle_timestamp = await self.get_latest_candle_timestamp()

        # Reset the last candle timestamp flag
        self.is_set_last_candle_timestamp = False
        self.log.info("Subscribing to %s websocket channels..", self.config.exchange)

        # Subscribe to candle updates for the specified asset list
        await self.ws_subscribe_candles(self.root.assets_list_symbols)

    async def ws_priv_init_connection(self):
        """Initialize the private websocket connection and subscribe to wallet updates."""
        self.log.info("Live Trading API initialized")

    async def ws_new_candle(self, candle: DictCandle):
        """
        Handle new candle updates.

        :param candle_obj: New candle object received from the websocket.
        :type candle_obj: dict
        """
        self.current_candle_timestamp = int(candle["mts"])
        self.update_candle_cache(candle)
        await self.process_candle_history_sync()
        await self.handle_new_candle()

    def update_candle_cache(self, candle: DictCandle):
        symbol_converted = convert_symbol_from_exchange(candle["symbol"])[0]
        current_asset = self.ws_candle_cache.get(self.current_candle_timestamp, {})
        current_asset[f"{symbol_converted}_o"] = candle["open"]
        current_asset[f"{symbol_converted}_c"] = candle["close"]
        current_asset[f"{symbol_converted}_h"] = candle["high"]
        current_asset[f"{symbol_converted}_l"] = candle["low"]
        current_asset[f"{symbol_converted}_v"] = candle["volume"]
        self.ws_candle_cache[self.current_candle_timestamp] = current_asset

    async def process_candle_history_sync(self):
        if not self.is_set_last_candle_timestamp:
            self.set_last_candle_timestamp()
            if self.should_patch_history():
                await self.patch_history()

    def set_last_candle_timestamp(self):
        candles_timestamps = self.ws_candle_cache.keys()
        candle_cache_size = len(candles_timestamps)
        if candle_cache_size >= 2:
            self.is_set_last_candle_timestamp = True
            self.last_candle_timestamp = max(candles_timestamps)
            self.log.debug("last_candle_timestamp set to %s", self.last_candle_timestamp)

    def should_patch_history(self):
        diff_range_candle_timestamps = check_timestamp_difference(
            self.log,
            start=self.latest_candle_timestamp,
            end=self.last_candle_timestamp,
            freq=self.config.candle_interval,
        )
        if len(diff_range_candle_timestamps) > 1:
            self.timestamp_patch_history_start = min(diff_range_candle_timestamps)
            self.timestamp_patch_history_end = max(diff_range_candle_timestamps)
            return self.timestamp_patch_history_start != self.timestamp_patch_history_end
        return False

    async def patch_history(self):
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
        """Handle new candle data received from the websocket."""
        if self.is_new_candle():
            self.prevent_race_condition_cache.append(self.current_candle_timestamp)
            await self.save_new_candle_to_db()
            if self.config.run_live:
                await self.trigger_trader_updates()

    def is_new_candle(self):
        """Returns True if the current candle is new and conditions are met to process it.

        Return True under following conditions:
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
        self.log.info("New candle received [timestamp: %s] - Saved to %s", self.last_candle_timestamp, self.config.dbms)
        candles_last_timestamp = self.ws_candle_cache.get(self.last_candle_timestamp, {})
        df_history_update = pd.DataFrame(candles_last_timestamp, index=[self.last_candle_timestamp])
        self.last_candle_timestamp = self.current_candle_timestamp
        df_history_update.index.name = "t"
        self.root.backend.db_add_history(df_history_update)
        self.prune_candle_cache()
        self.prune_race_condition_prevention_cache()

    async def trigger_trader_updates(self):
        if self.root.exchange_api.bfx_api_priv is not None:
            await self.root.trader.check_sold_orders()
        if not self.history_sync_patch_running and not self.config.is_sim:
            await self.root.trader.update()
            current_total_profit = self.root.trader.get_profit()
            self.log.info("Current total profit: $%s", current_total_profit)

    ##############################
    # Private websocket channels #
    ##############################

    # Handle order confirmed

    def ws_priv_order_confirmed(self, ws_confirmed):
        self.log.debug("order_confirmed: %s", str(ws_confirmed))
        asset_symbol = convert_symbol_from_exchange(ws_confirmed.symbol)[0]

        if get_order_type(ws_confirmed) == "sell":
            buy_order = {"asset": asset_symbol, "gid": ws_confirmed.gid}
            self.handle_sell_order(buy_order, ws_confirmed)
        else:
            self.log.error("Cannot find open order (%s)", str(buy_order))

    def handle_sell_order(self, buy_order, ws_confirmed):
        open_order = self.root.trader.get_open_order(asset=buy_order)
        if len(open_order) > 0:
            self.update_open_order(open_order[0], ws_confirmed)
        else:
            self.log.error("Cannot find open order (%s)", str(buy_order))

    def update_open_order(self, open_order, ws_confirmed):
        open_order["sell_order_id"] = ws_confirmed.id
        self.root.trader.edit_order(open_order, "open_orders")

    # Handle order closed

    async def ws_priv_order_closed(self, ws_order_closed):
        self.log.debug("order_closed: %s", str(ws_order_closed))
        if is_order_closed_and_filled(ws_order_closed):
            await self.handle_order_closed(ws_order_closed)

    async def handle_order_closed(self, ws_order_closed):
        order_type = get_order_type(ws_order_closed)
        if order_type == "buy":
            await buy_confirmed(self.root, ws_order_closed)
        if order_type == "sell":
            order_closed_dict = convert_order_to_dict(ws_order_closed)
            sell_confirmed(self.root, order_closed_dict)

    # Handle wallet updates

    async def ws_priv_wallet_snapshot(self, ws_wallet_snapshot):
        self.log.debug("wallet_snapshot: %s", str(ws_wallet_snapshot))
        self.root.trader.set_budget(ws_wallet_snapshot)
        self.root.backend.db_sync_state_trader()
        self.root.backend.db_sync_state_orders()
        await self.root.trader.get_min_order_sizes()

    def ws_priv_wallet_update(self, ws_wallet_update: Wallet):
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
