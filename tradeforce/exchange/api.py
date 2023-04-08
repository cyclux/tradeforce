"""_summary_

"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from tradeforce.utils import convert_symbol_from_exchange, convert_symbol_to_exchange, ns_to_ms

from bfxapi import Client  # type: ignore
from bfxapi.constants import WS_HOST, PUB_WS_HOST, REST_HOST, PUB_REST_HOST  # type: ignore
from tradeforce.trader.secure_credentials import load_credentials

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import Tradeforce
    from bfxapi.models import Order, Wallet  # type: ignore


class ExchangeAPI:
    """
    Provide an interface to interact with Bitfinex exchange APIs.
    """

    def __init__(self, root: Tradeforce):
        """
        Initialize the ExchangeAPI instance.
        Connect to the Bitfinex APIs.

        Args:
            root: The main Tradeforce instance. Provides access to other modules.
        """
        self.root = root
        self.config = root.config
        self.log = root.logging.get_logger(__name__)
        self.api_priv = self.connect_api_priv()
        self.api_pub = self.connect_api_pub()

    #########################
    # Connect Exchange APIs #
    #########################

    def connect_api_priv(self) -> Client:
        """
        Connect to Bitfinex private API.
        Private API requires credentials to be provided.
        Those are retrieved by the secure_credentials module.
        secure_credentials decrypts the stored credentials and provides them via context manager.
        The decrypted_credentials context manages to ensure that the credentials are deleted from memory after use.

        Returns:
            A Bitfinex Client instance or None if user configured not to run_live.
        """
        if not self.config.run_live:
            return None

        secure_credentials = load_credentials(self.root)
        with secure_credentials.decrypted_credentials() as credentials:
            bfx_api = Client(
                credentials["auth_key"],
                credentials["auth_sec"],
                ws_host=WS_HOST,
                rest_host=REST_HOST,
                logLevel=self.config.log_level_ws_live,
            )
        return bfx_api

    def connect_api_pub(self) -> Client:
        """
        Connect to Bitfinex public API.

        Returns:
            A Bitfinex Client instance or None if user configured not to set update_mode: once or live.
        """
        if self.config.update_mode not in ("once", "live"):
            return None

        bfx_api = Client(
            ws_host=PUB_WS_HOST,
            rest_host=PUB_REST_HOST,
            ws_capacity=25,
            max_retries=100,
            logLevel=self.config.log_level_ws_update,
        )
        return bfx_api

    ###############################
    # REST API endpoints - Public #
    ###############################

    async def get_active_assets(self, consider_exclude_list=True) -> list[str]:
        """
        Retrieve active / available assets from the exchange.

        Args:
            consider_exclude_list: If True, exclude assets listed in the config's assets_excluded list.

        Returns:
            A list of active assets symbols.
        """
        bfx_active_symbols = await self.api_pub.rest.fetch("conf/", params="pub:list:pair:exchange")
        symbols_base_currency = convert_symbol_from_exchange(bfx_active_symbols)
        if consider_exclude_list:
            assets_list_symbols: list[str] = np.setdiff1d(
                symbols_base_currency, self.config.assets_excluded, assume_unique=True
            ).tolist()
        else:
            assets_list_symbols = symbols_base_currency
        return assets_list_symbols

    async def get_asset_stats(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        """
        Retrieve asset statistics from the exchange.
        Those metrics will be helpful to determine the assets to inclulde in the portfolio.

        Args:
            symbols: A list of asset symbols e.g: ["BTC", "ETH", "XRP", "ADA", "LTC"].

        Returns:
            Dict of dicts: A dictionary containing asset statistics for each symbol.
        """
        bfx_asset_stats = await self.api_pub.rest.get_public_tickers(symbols)
        bfx_asset_stats_formatted = {}
        for asset_stat in bfx_asset_stats:
            bfx_asset_stats_formatted[asset_stat[0]] = {
                "BID": asset_stat[1],
                "BID_SIZE": asset_stat[2],
                "ASK": asset_stat[3],
                "ASK_SIZE": asset_stat[4],
                "DAILY_CHANGE": asset_stat[5],
                "DAILY_CHANGE_PERC": asset_stat[6],
                "LAST_PRICE": asset_stat[7],
                "VOLUME": asset_stat[8],
                "HIGH": asset_stat[9],
                "LOW": asset_stat[10],
            }
        return bfx_asset_stats_formatted

    async def get_public_candles(
        self,
        symbol: str,
        timestamp_start: int | None = None,
        timestamp_end: int | None = None,
        candle_type="hist",
    ) -> list:
        """
        Retrieve all of the public candles between the start and end period for a given symbol.

        Args:
            symbol:          The asset symbol e.g: "BTCUSD".
            timestamp_start: The start timestamp for the candle data.
            timestamp_end:   The end timestamp for the candle data.
            candle_type:     The type of candles to retrieve,
                                either "hist" for multiple candles or
                                or "last" for just the latest candle.

        Returns:
            A list of candle data for the given symbol.
        """
        symbol_bfx = convert_symbol_to_exchange(symbol)[0]

        candles = await self.api_pub.rest.get_public_candles(
            symbol_bfx, start=timestamp_start, end=timestamp_end, section=candle_type, limit=10000, tf="5m", sort=1
        )
        return candles

    async def get_latest_remote_candle_timestamp(self, minus_delta: str | None = None):
        """
        Retrieve the latest candle timestamp from the exchange.
        indicator_assets are popular assets on the exchange. This ensures that at least one of them
        is available and provides an up to date timestamp.

        We can utilize the latest timestamp as reference to determine valid timestamps in the past.
        By subtracting a time delta (e.g. 100days), we can retrieve valid timestamps as
        they get strictly incremented by the candle interval in milliseconds.
        e.g: 5 min candles are incremented by 300000 (5min * 60sec * 1000ms).

        Args:
            minus_delta: A string representing a time delta
                         to subtract from the latest candle timestamp.

        Returns:
            The latest remote candle timestamp.
        """
        indicator_assets = ["BTC", "ETH", "XRP", "ADA", "LTC"]

        latest_candle_timestamp_pool = []
        for indicator in indicator_assets:
            latest_exchange_candle = await self.get_public_candles(symbol=indicator, candle_type="last")
            latest_candle_timestamp_pool.append(int(latest_exchange_candle[0]))
        latest_candle_timestamp = max(latest_candle_timestamp_pool)

        if minus_delta:
            latest_candle = pd.Timestamp(latest_candle_timestamp, unit="ms", tz="UTC") - pd.to_timedelta(minus_delta)
            latest_candle_timestamp = ns_to_ms(int(latest_candle.value))
        return latest_candle_timestamp

    async def get_min_order_sizes(self) -> dict[str, float]:
        """
        Retrieve minimum order sizes for all assets.

        We can utilize this information to adapt the order size to the exchange's requirements.
        This is primarily used when placing small orders to test the trader.
        Symbols need to be in exchange format (e.g. "tBTCUSD" instead of "BTC") for the API call.
        Then converted back to the standard format (e.g. "BTC") for the return value.

        Returns:
            A dictionary containing the minimum order size for each asset symbol.
        """
        bfx_asset_infos = await self.api_pub.rest.fetch("conf/", params="pub:info:pair")
        asset_symbols = self.root.market_history.get_asset_symbols()
        all_asset_symbols = convert_symbol_to_exchange(asset_symbols, with_t_prefix=False)
        all_asset_symbols_info = [
            asset for asset in bfx_asset_infos[0] if asset[0][-3:] == "USD" and asset[0] in all_asset_symbols
        ]
        asset_min_order_sizes = {
            convert_symbol_from_exchange(asset[0])[0]: float(asset[1][3]) for asset in all_asset_symbols_info
        }
        return asset_min_order_sizes

    ################################
    # REST API endpoints - PRIVATE #
    ################################

    async def order(self, order_type: str, order_payload: dict) -> bool:
        """
        Submit an order to the exchange.
        Symbol needs to be in the "exchange format" e.g. "tBTCUSD".
        The amount needs to be positive for buy orders and negative for sell orders.
        "EXCHANGE FOK" is used for buy orders and "EXCHANGE LIMIT" for sell orders.
        FOK means "Fill or Kill" and will cancel the order if it cannot be filled immediately.
        LIMIT orders will be placed on the order book and only filled if the price is reached.

        Args:
            order_type:    The type of order, either "buy" or "sell".
            order_payload: A dictionary containing order details.

        Returns:
            True if the order was successfully submitted, otherwise False.
        """
        bfx_symbol = convert_symbol_to_exchange(order_payload["asset"])[0]
        order_amount = self.calculate_order_amount(order_type, order_payload["amount"])

        try:
            market_type_current = "EXCHANGE FOK" if order_type == "buy" else "EXCHANGE LIMIT"
            exchange_response = await self.api_priv.rest.submit_order(
                symbol=bfx_symbol,
                price=order_payload["price"],
                amount=order_amount,
                market_type=market_type_current,
                gid=order_payload["gid"],
            )
            return exchange_response.is_success()
        except Exception as exc:  # pylint: disable=broad-except
            self.log.exception(exc)
            return False

    async def edit_order(self, order_type: str, order_payload: dict) -> bool:
        """
        Update an existing order in the exchange.

        Args:
            order_type: The type of order, either "buy" or "sell".
            order_payload: A dictionary containing order details.

        Returns:
            True if the order was successfully updated, otherwise False.
        """
        order_amount = self.calculate_order_amount(order_type, order_payload["amount"])

        try:
            exchange_response = await self.api_priv.rest.submit_update_order(
                order_payload["sell_order_id"], price=order_payload["price"], amount=order_amount
            )
            return exchange_response.is_success()
        except Exception as exc:  # pylint: disable=broad-except
            self.log.exception(exc)
            return False

    def calculate_order_amount(self, order_type: str, amount: float) -> float:
        """
        Manipulate the order amount based on the order type.
        Positive amounts are used for buy orders, negative amounts for sell orders.

        Args:
            order_type: The type of order, either "buy" or "sell".
            amount:     The amount of the order.

        Returns:
            The calculated order amount.
        """
        return -1 * amount if order_type == "sell" else amount

    async def get_order_history(self, raw=False):
        """
        Retrieve order history from exchange.

        Args:
            raw: If True, return raw order history data directly from exchange.
                 If False, return a list of dictionaries containing relevant order items.

        Returns:
            A list of dictionaries containing order history data.
        """
        order_history = await self.api_priv.rest.post("auth/r/orders/hist", data={"limit": 2500}, params="")
        if not raw:
            order_history = [
                {
                    "id": order[0],
                    "gid": order[1],
                    "symbol": order[3],
                    "mts_create": order[4],
                    "mts_update": order[5],
                    "amount": order[6],
                    "amount_orig": order[7],
                    "type": order[8],
                    "order_status": order[13],
                    "price": order[16],
                    "price_avg": order[17],
                }
                for order in order_history
            ]
        return order_history

    async def get_active_orders(self, symbol: str) -> list[Order]:
        """
        Retrieve active orders for a given symbol from the exchange.

        Args:
            symbol: The asset symbol e.g. "BTCUSD".

        Returns:
            A list of active orders for the specified symbol. Order is a bitfinex specific class.
        """
        bfx_active_orders = await self.api_priv.rest.get_active_orders(symbol)
        return bfx_active_orders

    async def cancel_order(self, cid) -> bool:
        """
        Cancel an order on the exchange based on the order ID.

        Args:
            cid (int): The order ID.

        Returns:
            True if the order was successfully canceled, otherwise False.
        """
        bfx_cancel_order = await self.api_priv.rest.submit_cancel_order(cid)
        return bfx_cancel_order.is_success()

    async def get_wallets(self) -> list[Wallet]:
        """
        Retrieve wallet data from the exchange.
        Primary use case is to retrieve the available balance for a given asset.

        Returns:
            A list of Bitfinex specific Wallet objects.
        """
        bfx_wallets = await self.api_priv.rest.get_wallets()
        return bfx_wallets
