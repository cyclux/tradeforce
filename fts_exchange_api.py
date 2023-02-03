"""_summary_

Returns:
    _type_: _description_
"""

import numpy as np
from fts_utils import convert_symbol_str


class ExchangeAPI:
    """_summary_

    Returns:
        _type_: _description_
    """

    def __init__(self, fts_instance=None):
        self.fts_instance = fts_instance
        self.config = fts_instance.config

        self.bfx_api_priv = fts_instance.api["bfx_api_priv"]
        self.bfx_api_pub = fts_instance.api["bfx_api_pub"]

    #####################
    # REST API - Public #
    #####################

    async def get_active_assets(self, consider_exclude_list=True):
        bfx_active_symbols = await self.bfx_api_pub.rest.fetch("conf/", params="pub:list:pair:exchange")
        symbols_base_currency = convert_symbol_str(
            bfx_active_symbols,
            to_exchange=False,
            base_currency=self.config.base_currency,
            exchange=self.config.exchange,
        )
        if consider_exclude_list:
            assets_list_symbols = list(
                np.setdiff1d(symbols_base_currency, self.config.assets_excluded, assume_unique=True)
            )
        else:
            assets_list_symbols = symbols_base_currency
        return assets_list_symbols

    async def get_asset_stats(self, symbols):
        bfx_asset_stats = await self.bfx_api_pub.rest.get_public_tickers(symbols)
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
        self, symbol=None, base_currency=None, timestamp_start=None, timestamp_end=None, candle_type="hist"
    ):
        symbol_bfx = convert_symbol_str(
            symbol, to_exchange=True, base_currency=base_currency, exchange=self.config.exchange
        )

        candles = await self.bfx_api_pub.rest.get_public_candles(
            symbol_bfx, start=timestamp_start, end=timestamp_end, section=candle_type, limit=10000, tf="5m", sort=1
        )
        return candles

    ######################
    # REST API - PRIVATE #
    ######################

    async def order(self, order_type, order_payload, update_order=False):
        exchange_result = False
        bfx_symbol = convert_symbol_str(
            order_payload["asset"], base_currency=self.config.base_currency, to_exchange=True
        )
        if order_type == "sell":
            order_amount = -1 * order_payload["amount"]
        else:
            order_amount = order_payload["amount"]
        try:
            if update_order:
                # TODO: buy_order_id -> gid ?
                exchange_response = await self.bfx_api_priv.rest.submit_update_order(
                    order_payload["buy_order_id"], price=order_payload["price"], amount=order_amount
                )
            else:
                market_type_current = "EXCHANGE FOK" if order_type == "buy" else "EXCHANGE LIMIT"
                exchange_response = await self.bfx_api_priv.rest.submit_order(
                    symbol=bfx_symbol,
                    price=order_payload["price"],
                    amount=order_amount,
                    market_type=market_type_current,
                    gid=order_payload["gid"],
                )
            exchange_result = exchange_response.is_success()
        # Cannot specify Exception as bfx API does not provide more
        except Exception as exc:  # pylint: disable=broad-except
            print(exc)
        # else:
        #     if not update_order and order_type == "buy":
        #         self.gid_init += 1
        return exchange_result

    async def get_active_orders(self, symbol):
        bfx_active_orders = await self.bfx_api_priv.rest.get_active_orders(symbol)
        return bfx_active_orders

    async def cancel_order(self, cid):
        bfx_cancel_order = await self.bfx_api_priv.rest.submit_cancel_order(cid)
        return bfx_cancel_order

    async def get_wallets(self):
        bfx_wallets = await self.bfx_api_priv.rest.get_wallets()
        return bfx_wallets

    async def get_wallet_deposit_address(self, wallet=None, method=None, renew=0):
        bfx_deposit_address = await self.bfx_api_priv.rest.get_wallet_deposit_address(
            wallet=wallet, method=method, renew=renew
        )
        return bfx_deposit_address
