"""_summary_

Returns:
    _type_: _description_
"""
import asyncio
import sys
import numpy as np
import pandas as pd
from fts_utils import convert_symbol_str


def get_significant_digits(num, digits):
    return round(num, digits - int(np.floor(np.log10(abs(num)))) - 1)


def calc_fee(volume, exchange_fee, price_current, currency_type="crypto"):
    fee_to_pay = volume / 100 * exchange_fee
    volume_incl_fee = volume - fee_to_pay
    if currency_type == "crypto":
        amount_fee_fiat = np.round(fee_to_pay * price_current, 2)
    if currency_type == "fiat":
        amount_fee_fiat = fee_to_pay
    return volume_incl_fee, amount_fee_fiat


class Trader:
    """_summary_"""

    def __init__(self, fts_instance):
        self.fts_instance = fts_instance
        self.config = fts_instance.config

        self.config.backend = self.config.backend
        self.backend_client = self.fts_instance.backend.get_backend_client()
        self.backend_db = self.fts_instance.backend.get_backend_db()

        self.wallets = {}
        self.open_orders = []
        self.closed_orders = []
        self.min_order_sizes = {}
        self.gid = 10**9

        self.check_run_conditions()
        self.finalize_trading_config()
        # Only sync backend now if there is no exchange API connection.
        # In case an API connection is used, "sync_state_backend" will be called after API states have been received
        if self.config.use_backend and not self.config.run_exchange_api:
            self.sync_state_backend()

    def check_run_conditions(self):
        if (self.config.amount_invest_fiat is None) and (self.config.amount_invest_relative is None):
            sys.exit("[ERROR] Either 'amount_invest_fiat' or 'amount_invest_relative' must be set.")

        if (self.config.amount_invest_fiat is not None) and (self.config.amount_invest_relative is not None):
            print(
                "[INFO] 'amount_invest_fiat' and 'amount_invest_relative' are both set."
                + "'amount_invest_fiat' will be overwritten by 'amount_invest_relative' (relative to the budget)."
            )

    def finalize_trading_config(self):
        if self.config.amount_invest_relative is not None and self.config.budget > 0:
            self.config.amount_invest_fiat = float(np.round(self.config.budget * self.config.amount_invest_relative, 2))
        if self.config.buy_limit_strategy and self.config.budget > 0:
            self.config.asset_buy_limit = self.config.budget // self.config.amount_invest_fiat

    async def get_min_order_sizes(self):
        bfx_asset_infos = await self.fts_instance.exchange_api.bfx_api_pub.rest.fetch("conf/", params="pub:info:pair")
        asset_symbols = self.fts_instance.market_history.get_asset_symbols()
        all_asset_symbols = convert_symbol_str(
            asset_symbols, base_currency="USD", with_trade_prefix=False, to_exchange=True
        )
        all_asset_symbols_info = [
            asset for asset in bfx_asset_infos[0] if asset[0][-3:] == "USD" and asset[0] in all_asset_symbols
        ]
        asset_min_order_sizes = {
            convert_symbol_str(asset[0], to_exchange=False): float(asset[1][3]) for asset in all_asset_symbols_info
        }
        self.min_order_sizes = asset_min_order_sizes
        return asset_min_order_sizes

    def get_market_performance(self, history_window=150, timestamp=None):
        start = -1 * history_window
        end = None
        idx_type = "iloc"
        if timestamp is not None:
            idx_type = "loc"
            start = timestamp - (history_window * 300000)
            end = timestamp
        market_window_pct_change = self.fts_instance.market_history.get_market_history(
            start=start,
            end=end,
            idx_type=idx_type,
            pct_change=True,
            pct_as_factor=False,
            metrics=["o"],
            fill_na=True,
            uniform_cols=True,
        )
        if len(market_window_pct_change) < history_window:
            market_performance = None
        else:
            market_performance = market_window_pct_change.sum()  # .sort_values()
        return market_performance

    def check_buy_options(self, latest_prices=None, timestamp=None):
        if latest_prices is None:
            df_latest_prices = self.fts_instance.market_history.get_market_history(
                latest_candle=True, metrics=["o"], uniform_cols=True
            )
            latest_prices = df_latest_prices.to_dict("records")[0]
            if timestamp is None:
                timestamp = df_latest_prices.index[0]
        market_performance = self.get_market_performance(history_window=self.config.window, timestamp=timestamp)
        buy_options = []
        if market_performance is not None:
            buy_condition = (market_performance >= self.config.buy_opportunity_factor_min) & (
                market_performance <= self.config.buy_opportunity_factor_max
            )
            buy_options = market_performance[buy_condition]
            df_buy_options = pd.DataFrame({"perf": buy_options, "price": latest_prices}).dropna()
            if self.config.prefer_performance == "negative":
                df_buy_options = df_buy_options.sort_values(by="perf", ascending=True)
            if self.config.prefer_performance == "positive":
                df_buy_options = df_buy_options.sort_values(by="perf", ascending=False)
            if self.config.prefer_performance == "center":
                df_buy_options.loc[:, "perf"] = np.absolute(df_buy_options["perf"] - self.config.buy_opportunity_factor)
                df_buy_options = df_buy_options.sort_values(by="perf")

            df_buy_options.reset_index(names=["asset"], inplace=True)  # names=["symbol"]
            buy_options = df_buy_options.to_dict("records")

        return buy_options

    def check_sell_options(self, latest_prices=None, timestamp=None):
        sell_options = []
        if latest_prices is None:
            df_latest_prices = self.fts_instance.market_history.get_market_history(
                latest_candle=True, metrics=["c"], uniform_cols=True
            )
            timestamp = df_latest_prices.index[0]
            latest_prices = df_latest_prices.to_dict("records")[0]
        open_orders = self.get_all_open_orders(raw=True)
        for open_order in open_orders:
            price_current = latest_prices.get(open_order["asset"], 0)

            current_profit_ratio = price_current / open_order["price_buy"]
            time_since_buy = (timestamp - open_order["timestamp_buy"]) / 1000 / 60 // 5

            buy_orders_maxed_out = (
                len(open_orders) >= self.config.asset_buy_limit
                if self.config.buy_limit_strategy is True
                else self.config.budget < self.config.amount_invest_fiat
            )
            ok_to_sell = (
                time_since_buy > self.config.hold_time_limit
                and current_profit_ratio >= self.config.profit_ratio_limit
                and buy_orders_maxed_out
            )
            if self.config.is_simulation:
                if (price_current >= open_order["price_profit"]) or ok_to_sell:
                    # IF SIMULATION
                    # check plausibility and prevent false logic
                    # profit gets a max plausible threshold
                    if price_current / open_order["price_profit"] > 1.2:
                        price_current = open_order["price_profit"]
                    sell_option = {"asset": open_order["asset"], "price_sell": price_current}
                    sell_options.append(sell_option)
            else:
                if buy_orders_maxed_out and ok_to_sell:
                    sell_option = {
                        "asset": open_order["asset"],
                        "price_sell": price_current,
                        "gid": open_order["gid"],
                        "buy_order_id": open_order["buy_order_id"],
                    }
                    sell_options.append(sell_option)

        return sell_options

    def sync_state_backend(self):
        trader_id = self.config.trader_id
        db_acknowledged = False
        if self.config.backend == "mongodb":
            db_response = list(
                self.backend_db["trader_status"].find({"trader_id": trader_id}, projection={"_id": False})
            )
            if len(db_response) > 0 and db_response[0]["trader_id"] == trader_id:
                trader_status = db_response[0]
                if self.config.budget == 0:
                    self.config.budget = trader_status["budget"]
                    self.gid = trader_status["gid"]
                # TODO: Save remaining vals to DB
            else:
                trader_status = {
                    "trader_id": trader_id,
                    "window": self.config.window,
                    "budget": self.config.budget,
                    "buy_opportunity_factor": self.config.buy_opportunity_factor,
                    "buy_opportunity_boundary": self.config.buy_opportunity_boundary,
                    "profit_factor": self.config.profit_factor,
                    "amount_invest_fiat": self.config.amount_invest_fiat,
                    "exchange_fee": self.config.exchange_fee,
                    "gid": self.gid,
                }

                db_acknowledged = self.backend_db["trader_status"].insert_one(trader_status).acknowledged

            self.open_orders = list(
                self.backend_db["open_orders"].find({"trader_id": trader_id}, projection={"_id": False})
            )
            self.closed_orders = list(
                self.backend_db["closed_orders"].find({"trader_id": trader_id}, projection={"_id": False})
            )
        return db_acknowledged

    def backend_new_order(self, order, order_type):
        db_acknowledged = False
        if self.config.backend == "mongodb":
            db_acknowledged = self.backend_db[order_type].insert_one(order).acknowledged
        return db_acknowledged

    def backend_edit_order(self, order, order_type):
        db_acknowledged = False
        if self.config.backend == "mongodb":
            db_acknowledged = (
                self.backend_db[order_type]
                .update_one({"buy_order_id": order["buy_order_id"]}, {"$set": order})
                .acknowledged
            )
        return db_acknowledged

    def new_order(self, order, order_type):
        order_obj = getattr(self, order_type)
        order_obj.append(order)
        if self.config.use_backend:
            db_response = self.backend_new_order(order.copy(), order_type)
            if not db_response:
                print("[ERROR] Backend DB insert failed")

    def edit_order(self, order, order_type):
        order_obj = getattr(self, order_type)
        order_obj[:] = [o for o in order_obj if o.get("buy_order_id") != order["buy_order_id"]]
        order_obj.append(order)

        if self.config.use_backend:
            db_response = self.backend_edit_order(order.copy(), order_type)
            if not db_response:
                print("[ERROR] Backend DB insert failed")

    def backend_del_order(self, order, order_type):
        db_acknowledged = False
        if self.config.backend == "mongodb":
            db_acknowledged = (
                self.backend_db[order_type]
                .delete_one({"asset": order["asset"], "buy_order_id": order["buy_order_id"]})
                .acknowledged
            )
        return db_acknowledged

    def del_order(self, order, order_type):
        # Delete from internal mirror of DB
        order_obj = getattr(self, order_type)
        order_obj[:] = [o for o in order_obj if o.get("asset") != order["asset"]]
        if self.config.use_backend:
            db_response = self.backend_del_order(order.copy(), order_type)
            if not db_response:
                print("[ERROR] Backend DB delete order failed")

    def update_status(self, status_updates):
        db_acknowledged = False
        for status, value in status_updates.items():
            setattr(self, status, value)
        if self.config.use_backend:
            db_acknowledged = (
                self.backend_db["trader_status"]
                .update_one({"trader_id": self.config.trader_id}, {"$set": status_updates})
                .acknowledged
            )
        return db_acknowledged

    def get_open_order(self, asset_order=None, asset=None):
        gid = None
        buy_order_id = None
        price_profit = None
        if asset_order is not None:
            asset_symbol = convert_symbol_str(
                asset_order.symbol, base_currency=self.config.base_currency, to_exchange=False
            )
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

    async def buy_confirmed(self, buy_order):
        print("buy_order", buy_order)
        print("buy_order.symbol", buy_order.symbol)

        asset_price_profit = get_significant_digits(buy_order.price * self.config.profit_factor, 5)
        asset_symbol = convert_symbol_str(buy_order.symbol, base_currency=self.config.base_currency, to_exchange=False)
        buy_volume_fiat = float(np.round(self.config.amount_invest_fiat - buy_order.fee, 5))
        # Wait until balance is registered by websocket into self.wallets
        await asyncio.sleep(10)
        buy_volume_crypto = self.wallets[asset_symbol].balance_available
        open_order = {
            "trader_id": self.config.trader_id,
            "buy_order_id": buy_order.id,
            "gid": buy_order.gid,
            "timestamp_buy": int(buy_order.mts_create),
            "asset": asset_symbol,
            "base_currency": self.config.base_currency,
            # TODO: "performance": asset["perf"],
            "price_buy": buy_order.price,
            "price_profit": asset_price_profit,
            "amount_invest_fiat": self.config.amount_invest_fiat,
            "buy_volume_fiat": buy_volume_fiat,
            "buy_fee_fiat": buy_order.fee,
            "buy_volume_crypto": buy_volume_crypto,
        }
        self.new_order(open_order, "open_orders")
        if not self.config.is_simulation:
            await self.submit_sell_order(open_order)

    async def submit_sell_order(self, open_order):
        volatility_buffer = 0.00000002
        sell_order = {
            "asset": open_order["asset"],
            "price": open_order["price_profit"],
            "amount": open_order["buy_volume_crypto"] - volatility_buffer,
            "gid": open_order["gid"],
        }
        exchange_result_ok = await self.fts_instance.exchange_api.order("sell", sell_order)
        if not exchange_result_ok:
            print(f"[ERROR] Sell order execution failed! -> {sell_order}")

    def sell_confirmed(self, sell_order_object):
        print("sell_order", sell_order_object)
        asset_symbol = convert_symbol_str(
            sell_order_object.symbol, base_currency=self.config.base_currency, to_exchange=False
        )
        sell_order = {"asset": asset_symbol, "gid": sell_order_object.gid}
        open_order = self.get_open_order(asset=sell_order)
        if len(open_order) > 0:
            closed_order = open_order[0].copy()
            closed_order["sell_timestamp"] = sell_order_object.mts_update
            closed_order["price_sell"] = sell_order_object.price
            closed_order["sell_fee_fiat"] = float(np.round(sell_order_object.fee * sell_order_object.price, 2))
            closed_order["sell_volume_crypto"] = abs(sell_order_object.amount_orig)
            closed_order["sell_volume_fiat"] = float(
                np.round(abs(sell_order_object.amount_orig) * sell_order_object.price, 2)
            )
            closed_order["profit_fiat"] = closed_order["sell_volume_fiat"] - closed_order["amount_invest_fiat"]
            self.new_order(closed_order, "closed_orders")
            self.del_order(open_order[0], "open_orders")
        else:
            print(f"[ERROR] Could not find order to sell: {sell_order}")

    async def buy_assets(self, buy_options):
        compensate_rate_limit = bool(len(buy_options) > 9)
        for asset in buy_options:
            # TODO: Make possible to have multiple orders of same asset
            if asset in self.config.assets_excluded:
                print("[INFO] Asset on blacklist. Will not buy {asset}")
                continue
            asset_open_orders = self.get_open_order(asset=asset)
            if len(asset_open_orders) > 0:
                continue

            # Add 1% margin for BUY LIMIT order
            asset["price"] *= 1.015
            buy_volume_fiat, _ = calc_fee(
                self.config.amount_invest_fiat, self.config.exchange_fee, asset["price"], currency_type="fiat"
            )
            buy_amount_crypto = get_significant_digits(buy_volume_fiat / asset["price"], 9)

            asset_symbol = asset["asset"]
            min_order_size = self.min_order_sizes.get(asset_symbol, 0)
            if min_order_size > buy_amount_crypto:
                print(
                    f"[INFO] Adapting buy_amount_crypto ({buy_amount_crypto}) "
                    + f"of {asset_symbol} to min_order_size ({min_order_size})"
                )
                buy_amount_crypto = min_order_size * 1.02

            if self.config.budget < self.config.amount_invest_fiat:
                print(
                    f"[INFO] Out of funds to buy {asset_symbol}: "
                    f"${self.config.budget} < ${self.config.amount_invest_fiat}"
                )
                continue

            buy_order = {
                "asset": asset_symbol,
                "gid": self.gid,
                "price": asset["price"],
                "amount": buy_amount_crypto,
            }
            print("[INFO] Executing buy order:", buy_order)

            if self.config.is_simulation:
                new_budget = float(np.round(self.config.budget - self.config.amount_invest_fiat, 2))
                self.update_status({"budget": new_budget})
            else:
                exchange_result_ok = await self.fts_instance.exchange_api.order("buy", buy_order)
                self.gid += 1
                self.update_status({"gid": self.gid})
                if not exchange_result_ok:
                    # TODO: Send notification about this event!
                    print(f"[ERROR] Buy order execution failed! -> {buy_order}")
                if compensate_rate_limit:
                    await asyncio.sleep(0.8)

    async def sell_assets(self, sell_options):
        for sell_option in sell_options:
            open_order = self.get_open_order(asset=sell_option)
            if len(open_order) < 1:
                continue

            if self.config.is_simulation:
                closed_order = open_order[0].copy()
                closed_order["price_sell"] = sell_option["price_sell"]
                sell_volume_crypto, sell_fee_fiat = calc_fee(
                    closed_order["buy_volume_crypto"],
                    self.config.exchange_fee,
                    closed_order["price_sell"],
                    currency_type="crypto",
                )
                closed_order["sell_fee_fiat"] = sell_fee_fiat
                closed_order["sell_volume_crypto"] = sell_volume_crypto
                closed_order["sell_volume_fiat"] = sell_volume_crypto * closed_order["price_sell"]
                closed_order["profit_fiat"] = closed_order["sell_volume_fiat"] - closed_order["amount_invest_fiat"]
                self.new_order(closed_order, "closed_orders")
                self.del_order(open_order[0], "open_orders")

                new_budget = float(np.round(self.config.budget + closed_order["sell_volume_fiat"], 2))
                self.update_status({"budget": new_budget})
            else:
                # Adapt sell price
                volatility_buffer = 0.00000002
                sell_order = {
                    "sell_order_id": open_order[0]["sell_order_id"],
                    "gid": open_order[0]["gid"],
                    "asset": open_order[0]["asset"],
                    "price": sell_option["price_sell"],
                    "amount": open_order[0]["buy_volume_crypto"] - volatility_buffer,
                }
                order_result_ok = await self.fts_instance.exchange_api.order("sell", sell_order, update_order=True)
                if order_result_ok:
                    print(
                        f"[INFO] Sell price of {sell_order['asset']} has been changed "
                        + f"from {open_order[0]['price_buy']} to {sell_order['price']}."
                    )

    def get_all_open_orders(self, raw=False):
        if raw:
            all_open_orders = self.open_orders
        else:
            all_open_orders = pd.DataFrame(self.open_orders)
        return all_open_orders

    def get_all_closed_orders(self, raw=False):
        if raw:
            all_closed_orders = self.closed_orders
        else:
            all_closed_orders = pd.DataFrame(self.closed_orders)
        return all_closed_orders

    def set_budget(self, ws_wallet_snapshot):
        for wallet in ws_wallet_snapshot:
            if wallet.currency == self.config.base_currency:
                is_snapshot = wallet.balance_available is None
                base_currency_balance = wallet.balance if is_snapshot else wallet.balance_available
                self.config.budget = base_currency_balance
                self.update_status({"budget": base_currency_balance})

    async def update(self, latest_prices=None, timestamp=None):
        sell_options = self.check_sell_options(latest_prices, timestamp)
        if len(sell_options) > 0:
            await self.sell_assets(sell_options)
        buy_options = self.check_buy_options(latest_prices, timestamp)
        print("[DEBUG] Potential buy options:")
        print(buy_options)
        if len(buy_options) > 0:
            await self.buy_assets(buy_options)

    # TODO: Currently not used
    def get_profit(self):
        profit_fiat = sum(order["profit_fiat"] for order in self.closed_orders)
        return profit_fiat
