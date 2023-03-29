"""_summary_
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from psycopg2.sql import SQL, Identifier

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import TradingEngine
    from tradeforce.backend import BackendSQL


class CreateTables:
    def __init__(self, root: TradingEngine, backend: BackendSQL) -> None:
        self.root = root
        self.backend = backend
        self.config = root.config
        self.log = root.logging.get_logger(__name__)

    def history(self, asset_symbols) -> None:
        ohlcv = ("_o", "_h", "_l", "_c", "_v")
        asset_symbols_ohlcv = [symbol + suffix for symbol in asset_symbols for suffix in ohlcv]
        sql_columns = ", ".join(
            [
                f"{Identifier(column_name).as_string(self.backend.dbms_db)} NUMERIC"
                for column_name in asset_symbols_ohlcv
            ]
        )
        query = SQL(
            "CREATE TABLE {table_name} (" + "t BIGINT NOT NULL," + f"{sql_columns}," + "PRIMARY KEY (t)" + ");"
        ).format(table_name=Identifier(self.config.dbms_history_entity_name))
        self.backend.execute(query)
        self.log.info("Created table %s", self.config.dbms_history_entity_name)

    def trader_status(self) -> None:
        query = SQL(
            "CREATE TABLE {table_name} ("
            + "trader_id INTEGER NOT NULL,"
            + "gid BIGINT NOT NULL,"
            + "moving_window_increments BIGINT NOT NULL,"
            + "budget NUMERIC NOT NULL,"
            + "buy_opportunity_factor NUMERIC NOT NULL,"
            + "buy_opportunity_boundary NUMERIC NOT NULL,"
            + "profit_factor NUMERIC NOT NULL,"
            + "amount_invest_fiat NUMERIC NOT NULL,"
            + "maker_fee NUMERIC NOT NULL,"
            + "taker_fee NUMERIC NOT NULL,"
            + "PRIMARY KEY (gid)"
            + ");"
        ).format(table_name=Identifier("trader_status"))
        self.backend.execute(query)
        self.log.info("Created table %s", "trader_status")

    def open_orders(self) -> None:
        query = SQL(
            "CREATE TABLE {table_name} ("
            + "trader_id INTEGER NOT NULL,"
            + "gid BIGINT NOT NULL,"
            + "buy_order_id BIGINT NOT NULL,"
            + "timestamp_buy BIGINT NOT NULL,"
            + "asset VARCHAR(10) NOT NULL,"
            + "base_currency VARCHAR(10) NOT NULL,"
            + "price_buy NUMERIC(10,8) NOT NULL,"
            + "price_profit NUMERIC(10,8) NOT NULL,"
            + "amount_invest_fiat NUMERIC(10,2) NOT NULL,"
            + "buy_volume_fiat NUMERIC(10,2) NOT NULL,"
            + "buy_fee_fiat NUMERIC(5,12) NOT NULL,"
            + "buy_volume_crypto NUMERIC(10,16) NOT NULL,"
            + "sell_order_id BIGINT,"
            + "PRIMARY KEY (gid)"
            + ");"
        ).format(table_name=Identifier("open_orders"))
        self.backend.execute(query)
        self.log.info("Created table %s", "open_orders")

    def closed_orders(self) -> None:
        query = SQL(
            "CREATE TABLE {table_name} ("
            + "trader_id INTEGER NOT NULL,"
            + "gid BIGINT NOT NULL,"
            + "buy_order_id BIGINT NOT NULL,"
            + "timestamp_buy BIGINT NOT NULL,"
            + "asset VARCHAR(10) NOT NULL,"
            + "base_currency VARCHAR(10) NOT NULL,"
            + "price_buy NUMERIC(10,8) NOT NULL,"
            + "price_profit NUMERIC(10,8) NOT NULL,"
            + "amount_invest_fiat NUMERIC(10,2) NOT NULL,"
            + "buy_volume_fiat NUMERIC(10,2) NOT NULL,"
            + "buy_fee_fiat NUMERIC(5,12) NOT NULL,"
            + "buy_volume_crypto NUMERIC(10,16) NOT NULL,"
            + "sell_order_id BIGINT,"
            + "timestamp_sell BIGINT,"
            + "price_sell NUMERIC(10,8) NOT NULL,"
            + "sell_volume_crypto NUMERIC(10,16) NOT NULL,"
            + "sell_fee_fiat NUMERIC(5,12) NOT NULL,"
            + "sell_volume_fiat NUMERIC(10,2) NOT NULL,"
            + "profit_fiat NUMERIC(10,2) NOT NULL,"
            + "PRIMARY KEY (gid)"
            + ");"
        ).format(table_name=Identifier("closed_orders"))
        self.backend.execute(query)
        self.log.info("Created table %s", "closed_orders")
