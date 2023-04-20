""" trader/trader_core.py

Module: tradeforce.trader
-------------------------

Trader module manages live trading on an exchange through the Trader class.

It uses a combination of in-memory and database storage to handle
open and closed orders, and offers functionality to update the trader
with the latest prices, process buy/sell options, and check for sold orders.

Relies on external modules such as numpy, pandas, and bfxapi
for various operations. It also uses other internal components, such as
the check_buy_options, buy_assets, check_sell_options, sell_assets,
and sell_confirmed functions, for making trading decisions and executing orders.

Contains a few utility functions, such as _build_query, _filter_sold_orders,
and _get_base_currency_balance, which assist in various tasks like
building query strings, filtering out sold orders, and
retrieving base currency balances, respectively.

The Trader class manages the order history, budgets, and wallets.
It provides methods to add, edit, or delete orders, retrieve
open or closed orders, update the trader's budget, and calculate profit.

Main Methods:

    new_order:
    Adds a new order to the list of orders, performing the operation on both
        the in-memory list of orders and the database if available.

    edit_order:
    Edits an existing order in the list of orders, updating the in-memory list
        of orders and the database if available.

    del_order:
    Deletes an order from the list of orders, performing the operation on both
        the in-memory list of orders and the database if available.

    get_open_order:
    Retrieves the open orders for a specific asset based on the provided asset details.

    get_all_open_orders:
    Returns a list of dictionaries containing all open orders.

    get_all_closed_orders:
    Returns a list of dictionaries containing all closed orders.

    candles_update:
    Updates the trader with the latest prices (candles) and processes
        buy and sell options based on the current market conditions.

    check_sold_orders:
    Checks for sold orders and confirms them, converting them to closed orders
        by logging relevant information and confirming the sale.

    set_budget:
    Sets the trader's budget based on the available balance of the base currency wallet.

    get_profit:
    Calculates the profit from closed orders.

    get_min_order_sizes:
    Retrieves the minimum order sizes for assets from the exchange
        and stores them in the Trader instance.

"""


from __future__ import annotations
import numpy as np
import pandas as pd
from bfxapi.models.wallet import Wallet  # type: ignore
from typing import TYPE_CHECKING
from tradeforce.trader.buys import check_buy_options, buy_assets
from tradeforce.trader.sells import check_sell_options, sell_assets, sell_confirmed

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import Tradeforce


def _build_query(asset: dict) -> str:
    """Build a query string for searching assets in the open_orders DataFrame.

    Params:
        asset: Dict containing the asset details.

    Returns:
        A string representing the query that can be used with DataFrame.query().
    """
    query_parts = [f"asset == '{asset['asset']}'"]

    if asset.get("buy_order_id", None):
        query_parts.append(f"buy_order_id == {asset['buy_order_id']}")
    if asset.get("gid", None):
        query_parts.append(f"gid == {asset['gid']}")
    if asset.get("price_profit", None):
        query_parts.append(f"price_profit == {asset['price_profit']}")

    return " and ".join(query_parts)


def _filter_sold_orders(exchange_order_history: list[dict], sell_order_ids: list[int]) -> list[dict]:
    """Filter out the sold orders from the given order history.

    Params:
        exchange_order_history: A list of dictionaries containing the order history.
        sell_order_ids:         A list of sell order IDs, returned by the exchange.

    Returns:
        A list of dictionaries containing sold orders.
    """
    return [
        order
        for order in exchange_order_history
        if order["id"] in sell_order_ids and "EXECUTED" in order["order_status"]
    ]


def _get_base_currency_balance(wallet: Wallet) -> float:
    """Retrieve the base currency balance from a Wallet object.

    Determine if the Wallet object is from a snapshot or regular Wallet update.
    Regular Wallet updates contain the balance_available attribute.
    Snapshot Wallet updates instead contain the balance attribute.

    Params:
        wallet: A Wallet object containing balance details.

    Returns:
        A float representing the base currency balance.
    """
    is_snapshot = wallet.balance_available is None
    return wallet.balance if is_snapshot else wallet.balance_available


class Trader:
    """The Trader class is responsible for managing and processing buy and sell orders.

    It also manages order history and provides functionality to check for sold orders.
    """

    def __init__(self, root: Tradeforce):
        self.root = root
        self.config = root.config
        self.log = root.logging.get_logger(__name__)

        self.wallets: dict[str, Wallet] = {}
        self.open_orders: list[dict] = []
        self.closed_orders: list[dict] = []
        self.min_order_sizes: dict[str, float] = {}
        self.gid = 10**9  # initial gid for orders

    # -----------------#
    # Order operations #
    # ------------------#

    def _handle_db_operations(self, action: str, order: dict, order_type: str) -> None:
        """Perform database operations to insert, edit, or delete an order.

        Params:
            action:     A string indicating the type of database operation to perform,
                        must be one of 'new', 'edit', or 'delete'.
            order:      Dict containing the order details.
            order_type: A string representing the type of order, e.g., 'open_orders' or 'closed_orders'.
        """
        if not self.config.use_dbms:
            return

        if action == "new":
            db_response = self.root.backend.insert_one(order_type, order.copy())
            if not db_response:
                self.log.error("Backend DB insert order failed!")
        elif action == "edit":
            update_ok = self.root.backend.update_one(
                order_type, query={"attribute": "buy_order_id", "value": order["buy_order_id"]}, set_value=order
            )
            if not update_ok:
                self.log.error("Backend DB edit order failed!")
        elif action == "delete":
            delete_ok = self.root.backend.delete_one(
                order_type, query={"attribute": "buy_order_id", "value": order["buy_order_id"]}
            )
            if not delete_ok:
                self.log.error("Backend DB delete order failed!")
        else:
            raise ValueError("Invalid action provided. Must be one of 'new', 'edit', or 'delete'.")

    def new_order(self, order: dict, order_type: str) -> None:
        """Add a new order to the list of orders.

        Performs this operation on the in memory list of orders and the DB if available.

        Params:
            order:      Dict containing the order details.
            order_type: A string representing the type of order, e.g., 'open_orders' or 'closed_orders'.
        """
        order_obj: list[dict] = getattr(self, order_type)
        order_obj.append(order)
        self._handle_db_operations("new", order, order_type)

    def edit_order(self, order: dict, order_type: str) -> None:
        """Edit an existing order in the list of orders.

        Performs this operation on the in memory list of orders and the DB if available.

        Params:
            order:      Dict containing the updated order details.
            order_type: A string representing the type of order, e.g., 'open_orders' or 'closed_orders'.
        """
        order_obj: list[dict] = getattr(self, order_type)
        order_obj[:] = [o for o in order_obj if o.get("buy_order_id") != order["buy_order_id"]]
        order_obj.append(order)
        self._handle_db_operations("edit", order, order_type)

    def del_order(self, order: dict, order_type: str) -> None:
        """Delete an order from the list of orders.

        Performs this operation on the in-memory list of orders and the DB if available.

        Params:
            order:      Dict containing the order details to be deleted.
            order_type: A string representing the type of order, e.g., 'open_orders' or 'closed_orders'.
        """
        order_obj: list[dict] = getattr(self, order_type)
        order_obj[:] = [o for o in order_obj if o.get("asset") != order["asset"]]
        self._handle_db_operations("delete", order, order_type)

    # ----------------#
    # Getting orders #
    # ----------------#

    def get_open_order(self, asset: dict) -> list[dict]:
        """Get the open orders for a specific asset based on the provided asset details.

        Params:
            asset: Dict containing the asset details.

        Returns:
            A list of dictionaries containing the open orders for the specified asset.
        """
        open_orders = []
        query = _build_query(asset)
        df_open_orders = pd.DataFrame(self.open_orders)

        if not df_open_orders.empty:
            open_orders = df_open_orders.query(query).to_dict("records")

        return open_orders

    def get_all_open_orders(self) -> list[dict]:
        """Get all open orders.

        Returns:
            A list of dictionaries containing all open orders.
        """
        return self.open_orders

    def get_all_closed_orders(self) -> list[dict]:
        """Get all closed orders.

        Returns:
            A list of dictionaries containing all closed orders.
        """
        return self.closed_orders

    # ---------------------------#
    # Trader relevant functions #
    # ---------------------------#

    async def candles_update(self) -> None:
        """Update the trader with the latest prices (candles) and process buy/sell options.

        If a timestamp is provided, the trader will check if it is time to sell assets
        based on the elapsed time since buy.

        Params:
            latest_prices: Dict containing the latest price data.
            timestamp: An integer representing the current timestamp.
        """
        sell_options = check_sell_options(self.root)

        if sell_options:
            await sell_assets(self.root, sell_options)

        buy_options = check_buy_options(self.root)
        if buy_options:
            await buy_assets(self.root, buy_options)

    async def check_sold_orders(self) -> None:
        """Check for sold orders and confirm them, converting them to closed orders.

        Retrieves the order history from the exchange and filters out the sold orders.
        For each sold order, log the relevant information and confirm the sale.
        """
        open_orders = pd.DataFrame(self.get_all_open_orders())

        if open_orders.empty:
            return

        sell_order_ids = open_orders["sell_order_id"].to_list()
        exchange_order_history = await self.root.exchange_api.get_order_history()
        sold_orders = _filter_sold_orders(exchange_order_history, sell_order_ids)

        for sold_order in sold_orders:
            self.log.info(
                "Sold order of %s (id:%s gid:%s) has been converted to closed order",
                sold_order["symbol"],
                sold_order["id"],
                sold_order["gid"],
            )
            sell_confirmed(self.root, sold_order)

    def _find_base_currency_wallet(self, ws_wallet_snapshot: list[Wallet]) -> Wallet | None:
        """Find the base currency wallet from a list of Wallet objects.

        Its amount is equivalent to the current available budget.

        Params:
            ws_wallet_snapshot: A list of Wallet objects.

        Returns:
            A Wallet object representing the base currency wallet, or None if not found.
        """
        for wallet in ws_wallet_snapshot:
            if wallet.currency == self.config.base_currency:
                return wallet
        return None

    def set_budget(self, ws_wallet_snapshot: list[Wallet]) -> None:
        """Set the trader's budget

        based on the available balance of the base currency wallet.
        Retrieve the base currency wallet and update the budget with its available balance.

        Params:
            ws_wallet_snapshot: A list of Wallet objects.
        """
        base_currency_wallet = self._find_base_currency_wallet(ws_wallet_snapshot)

        if base_currency_wallet is not None:
            base_currency_balance = _get_base_currency_balance(base_currency_wallet)
            self.config.budget = base_currency_balance
            self.root.backend.update_status({"budget": base_currency_balance})

    def get_profit(self) -> float:
        """Calculate the profit from closed orders.

        Returns:
            A float representing the total profit.
        """
        return np.round(sum(order["profit_fiat"] for order in self.closed_orders), 2)

    async def get_min_order_sizes(self) -> None:
        """Retrieve the minimum order sizes

        for assets from the exchange and store them in the Trader instance.
        """
        self.min_order_sizes = await self.root.exchange_api.get_min_order_sizes()
