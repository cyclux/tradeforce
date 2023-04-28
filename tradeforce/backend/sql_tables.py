""" backend/sql_tables.py

Module: tradeforce.backend.sql_tables
-------------------------------------

Contains the CreateTables class for creating and initializing tables in the Postgres DB.
It provides methods for creating and initializing tables that store various types of data,
such as historical market data, trader status, open orders, and closed orders.

"""

from __future__ import annotations
from typing import TYPE_CHECKING
from psycopg2.sql import SQL, Identifier

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import Tradeforce
    from tradeforce.backend import BackendSQL
    from psycopg2.extensions import cursor


class CreateTables:
    """Handles the creation and initialization of tables in the PostgreSQL database.

    Provides methods for creating tables in a PostgreSQL database to store
    various types of data, such as historical market data, trader status,
    open orders, and closed orders.
    """

    def __init__(self, root: Tradeforce, backend: BackendSQL) -> None:
        self.root = root
        self.backend = backend
        self.config = root.config
        self.log = root.logging.get_logger(__name__)

    def history(self, asset_symbols: list[str]) -> None:
        """Create and initialize the history table
        -> for storing historical market data.

        Create the history table with columns for each asset symbol
        and its OHLCV values. Do not create the table if it already exists.

        Params:
            asset_symbols: A list of asset symbols for which historical
                            market data will be stored.

        Raises:
            ValueError: If no asset symbols are provided.
        """
        # Do not create the table if it already exists
        if not self.backend.is_new_history_entity:
            return

        if not asset_symbols:
            raise ValueError(f"Cannot create {self.config.dbms_history_entity_name} table: No asset_symbols provided!")

        # Append suffixes to asset symbols to create column names
        ohlcv = ("_o", "_h", "_l", "_c", "_v")
        asset_symbols_ohlcv = [symbol + suffix for symbol in asset_symbols for suffix in ohlcv]

        psycopg2_cursor: cursor = self.backend.dbms_db
        sql_columns = ", ".join(
            [f"{Identifier(column_name).as_string(psycopg2_cursor)} NUMERIC" for column_name in asset_symbols_ohlcv]
        )

        query = SQL(
            "CREATE TABLE {table_name} (" + "t BIGINT NOT NULL," + f"{sql_columns}," + "PRIMARY KEY (t)" + ");"
        ).format(table_name=Identifier(self.config.dbms_history_entity_name))
        self.backend.execute(query)

        self.backend.is_new_history_entity = False

        self.log.info("Created table %s", self.config.dbms_history_entity_name)

    def trader_status(self) -> None:
        """Create and initialize the trader_status table
        -> for storing trader status data.

        Create the 'trader_status' table with columns
        for various trader-related data.
        """
        query = SQL(
            "CREATE TABLE {table_name} ("
            + "trader_id INTEGER NOT NULL,"
            + "gid BIGINT NOT NULL,"
            + "moving_window_increments BIGINT NOT NULL,"
            + "budget NUMERIC NOT NULL,"
            + "buy_signal_score NUMERIC NOT NULL,"
            + "buy_signal_boundary NUMERIC NOT NULL,"
            + "profit_factor_target NUMERIC NOT NULL,"
            + "amount_invest_per_asset NUMERIC NOT NULL,"
            + "maker_fee NUMERIC NOT NULL,"
            + "taker_fee NUMERIC NOT NULL,"
            + "PRIMARY KEY (gid)"
            + ");"
        ).format(table_name=Identifier("trader_status"))
        self.backend.execute(query)

        self.log.info("Created table %s", "trader_status")

    def open_orders(self) -> None:
        """Create and initialize the open_orders table
        -> for storing open orders data.

        Create the 'open_orders' table with columns
        for various order-related data for open orders.
        """
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
            + "amount_invest_per_asset NUMERIC(10,2) NOT NULL,"
            + "buy_volume_fiat NUMERIC(10,2) NOT NULL,"
            + "buy_fee_fiat NUMERIC(5,12) NOT NULL,"
            + "buy_volume_asset NUMERIC(10,16) NOT NULL,"
            + "sell_order_id BIGINT,"
            + "PRIMARY KEY (gid)"
            + ");"
        ).format(table_name=Identifier("open_orders"))
        self.backend.execute(query)

        self.log.info("Created table %s", "open_orders")

    def closed_orders(self) -> None:
        """Creat and initialize the closed_orders table
        -> for storing closed orders data.

        Creates the 'closed_orders' table with columns
        for various order-related data for closed orders.
        """
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
            + "amount_invest_per_asset NUMERIC(10,2) NOT NULL,"
            + "buy_volume_fiat NUMERIC(10,2) NOT NULL,"
            + "buy_fee_fiat NUMERIC(5,12) NOT NULL,"
            + "buy_volume_asset NUMERIC(10,16) NOT NULL,"
            + "sell_order_id BIGINT,"
            + "timestamp_sell BIGINT,"
            + "price_sell NUMERIC(10,8) NOT NULL,"
            + "sell_volume_asset NUMERIC(10,16) NOT NULL,"
            + "sell_fee_fiat NUMERIC(5,12) NOT NULL,"
            + "sell_volume_fiat NUMERIC(10,2) NOT NULL,"
            + "profit_fiat NUMERIC(10,2) NOT NULL,"
            + "PRIMARY KEY (gid)"
            + ");"
        ).format(table_name=Identifier("closed_orders"))
        self.backend.execute(query)

        self.log.info("Created table %s", "closed_orders")
