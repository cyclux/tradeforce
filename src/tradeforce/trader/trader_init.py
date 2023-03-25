"""_summary_

Returns:
    _type_: _description_
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from bfxapi.models.wallet import Wallet  # type: ignore
from typing import TYPE_CHECKING
from tradeforce.utils import convert_symbol_from_exchange
from tradeforce.trader.buys import check_buy_options, buy_assets
from tradeforce.trader.sells import check_sell_options, sell_assets, sell_confirmed

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import TradingEngine
# TODO: switch order["buy_order_id"] to order["gid"]


class Trader:
    """_summary_"""

    def __init__(self, root: TradingEngine):
        self.root = root
        self.config = root.config
        self.log = root.logging.getLogger(__name__)

        self.wallets: dict[str, Wallet] = {}
        self.open_orders: list[dict] = []
        self.closed_orders: list[dict] = []
        self.min_order_sizes: dict[str, float] = {}
        self.gid = 10**9

    ####################
    # Order operations #
    ####################

    def new_order(self, order: dict, order_type: str) -> None:
        order_obj = getattr(self, order_type)
        order_obj.append(order)
        if self.config.use_dbms:
            db_response = self.root.backend.insert_one(order.copy(), order_type)
            if not db_response:
                self.log.error("Backend DB insert order failed!")
            # db_response = self.root.backend.order_new(order.copy(), order_type)

    def edit_order(self, order: dict, order_type: str) -> None:
        order_obj = getattr(self, order_type)
        order_obj[:] = [o for o in order_obj if o.get("buy_order_id") != order["buy_order_id"]]
        order_obj.append(order)

        if self.config.use_dbms:
            update_ok = self.root.backend.update_one(
                order_type, query={"attribute": "buy_order_id", "value": order["buy_order_id"]}, set_value=order
            )
            if not update_ok:
                self.log.error("Backend DB edit order failed!")
            # db_response = self.root.backend.order_edit(order.copy(), order_type)

    def del_order(self, order: dict, order_type: str) -> None:
        # Delete from internal mirror of DB
        order_obj = getattr(self, order_type)
        order_obj[:] = [o for o in order_obj if o.get("asset") != order["asset"]]
        if self.config.use_dbms:
            delete_ok = self.root.backend.delete_one(
                order_type, query={"attribute": "buy_order_id", "value": order["buy_order_id"]}
            )
            if not delete_ok:
                self.log.error("Backend DB delete order failed!")
            # delete_ok = self.root.backend.order_del(order.copy(), order_type)

    ##################
    # Getting orders #
    ##################

    def get_open_order(self, asset_order=None, asset=None) -> list:
        gid = None
        buy_order_id = None
        price_profit = None
        if asset_order is not None:
            asset_symbol = convert_symbol_from_exchange(asset_order.symbol)[0]
            buy_order_id = asset_order.id
            gid = asset_order.gid
        else:
            asset_symbol = asset["asset"]
            buy_order_id = asset.get("buy_order_id", None)
            gid = asset.get("gid", None)
            price_profit = asset.get("price_profit", None)

        query = f"asset == '{asset_symbol}'"

        if buy_order_id is not None:
            query += f" and buy_order_id == {buy_order_id}"
        if gid is not None:
            query += f" and gid == {gid}"
        if price_profit is not None:
            query += f" and price_profit == {price_profit}"

        open_orders = []
        df_open_orders = pd.DataFrame(self.open_orders)
        if not df_open_orders.empty:
            open_orders = df_open_orders.query(query).to_dict("records")
        return open_orders

    def get_all_open_orders(self) -> list:
        return self.open_orders

    def get_all_closed_orders(self, raw=False):
        if raw:
            all_closed_orders = self.closed_orders
        else:
            all_closed_orders = pd.DataFrame(self.closed_orders)
        return all_closed_orders

    #############################
    # Trader relevant functions #
    #############################

    async def update(self, latest_prices=None, timestamp=None):
        sell_options = check_sell_options(self.root, latest_prices, timestamp)
        if len(sell_options) > 0:
            await sell_assets(self.root, sell_options)
        buy_options = check_buy_options(self.root, latest_prices, timestamp)
        if len(buy_options) > 0:
            await buy_assets(self.root, buy_options)

    async def check_sold_orders(self) -> None:
        open_orders = pd.DataFrame(self.get_all_open_orders())
        if open_orders.empty:
            return
        sell_order_ids = open_orders["sell_order_id"].to_list()
        exchange_order_history = await self.root.exchange_api.get_order_history()
        sold_orders = [
            order
            for order in exchange_order_history
            if order["id"] in sell_order_ids and "EXECUTED" in order["order_status"]
        ]
        for sold_order in sold_orders:
            self.log.info(
                "Sold order of %s (id:%s gid:%s) has been converted to closed order",
                sold_order["symbol"],
                sold_order["id"],
                sold_order["gid"],
            )
            sell_confirmed(self.root, sold_order)

    def set_budget(self, ws_wallet_snapshot):
        for wallet in ws_wallet_snapshot:
            if wallet.currency == self.config.base_currency:
                is_snapshot = wallet.balance_available is None
                base_currency_balance = wallet.balance if is_snapshot else wallet.balance_available
                self.config.budget = base_currency_balance
                self.root.backend.update_status({"budget": base_currency_balance})

    def get_profit(self):
        profit_fiat = np.round(sum(order["profit_fiat"] for order in self.closed_orders), 2)
        return profit_fiat

    async def get_min_order_sizes(self):
        self.min_order_sizes = await self.root.exchange_api.get_min_order_sizes()
