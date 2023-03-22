"""_summary_

Returns:
    _type_: _description_
"""
from __future__ import annotations
import sys

# import pandas as pd
from psycopg2.extensions import connection, cursor
from psycopg2 import OperationalError
import psycopg2.pool as pool
from psycopg2.sql import SQL, Composed, Identifier, Literal
from typing import TYPE_CHECKING

# from urllib.parse import urlparse
from tradeforce.backend import Backend
from tradeforce.backend.sql_tables import CreateTables


# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import TradingEngine
# FIXME sell_timestamp -> timestamp_sell


class BackendSQL(Backend):
    def __init__(self, root: TradingEngine):
        super().__init__(root)
        self.connected = False
        self.create_table = CreateTables(root, self)
        self.init_connect(db_name=self.config.dbms_db)
        # Only sync backend now if there is no exchange API connection.
        # In case an API connection is used, db_sync_trader_state()
        # will be called once by exchange_ws -> ws_priv_wallet_snapshot()
        if self.config.use_dbms and not self.config.run_live:
            self.db_sync_trader_state()

    def connect(self, db_name) -> None:
        dbms_uri = self.construct_uri(db_name)
        try:
            self.pool = pool.SimpleConnectionPool(minconn=1, maxconn=10, dsn=dbms_uri)
        except OperationalError:
            self.log.info("Failed to connect to database %s. Trying again...", db_name)
            self.connected = False
            pass
        else:
            self.connected = True

    def init_connect(self, db_name=None) -> None:
        self.connect(db_name)
        if self.connected:
            pool_connection: connection = self.pool.getconn()
            with pool_connection as dbms_client:
                dbms_client.autocommit = True
                self.dbms_db: cursor = dbms_client.cursor()

            if db_name == self.config.dbms_connect_db:
                self.db_exists_or_create()
                self.dbms_db.close()
                self.init_connect(self.config.dbms_db)
            if db_name == self.config.dbms_db:
                self.is_new_coll_or_table = self.check_table()
                if self.is_new_coll_or_table:
                    self.create_table.trader_status()
                    self.create_table.open_orders()
                    self.create_table.closed_orders()

        else:
            self.init_connect(self.config.dbms_connect_db)

    def execute(self, query: Composed):
        try:
            self.dbms_db.execute(query)
            execute_ok = True
        except (TypeError, OperationalError):
            execute_ok = False
            self.log.error("SQL execute failed!")
        return execute_ok

    def db_exists_or_create(self):
        query = SQL("SELECT 1 FROM pg_catalog.pg_database WHERE datname = {db_name};").format(
            db_name=Literal(self.config.dbms_db)
        )
        self.execute(query)
        db_exists = self.dbms_db.fetchone()
        if not db_exists:
            query = SQL("CREATE DATABASE {db_name};").format(db_name=Identifier(self.config.dbms_db))
            self.execute(query)
            self.log.info("Created database %s", self.config.dbms_db)
        else:
            self.log.info("Database %s already exists", self.config.dbms_db)

    def check_table(self):
        query = SQL("SELECT 1 FROM pg_catalog.pg_tables WHERE tablename = {table_name};").format(
            table_name=Literal(self.config.dbms_table_or_coll_name)
        )
        self.execute(query)
        table_exists = self.dbms_db.fetchone()
        if not table_exists:
            is_new_table = True
        else:
            is_new_table = False
            if self.config.force_source != "postgresql":
                if self.config.force_source == "api":
                    sys.exit(
                        f"[ERROR] PostgreSQL history '{self.config.dbms_table_or_coll_name}' already exists. "
                        + "Cannot load history via API. Choose different history DB name or loading method."
                    )
                self.sync_check_needed = True
        return is_new_table

        #########################
        # PostgreSQL operations #
        #########################

    def create_index(self, table_name, index_name, unique=False) -> None:
        query = SQL("CREATE {unique} INDEX {index_name} ON {table_name} ({index_name});").format(
            unique=SQL("UNIQUE") if unique else SQL(""),
            index_name=Identifier(index_name),
            table_name=Identifier(table_name),
        )
        self.execute(query)
        self.log.info("Created index %s on %s", index_name, table_name)

    def query(self, table_name, query=None, projection=None, sort=None, limit=None, skip=None) -> list:
        # query={"attribute": "buy_order_id", "value": order["buy_order_id"]}
        # SELECT * FROM table WHERE status IN ('Open', 'In Progress')

        # self.query("trader_status", query={"attribute": "trader_id", "value": trader_id})
        #         external_db_index = self.query(  # type: ignore
        # self.config.dbms_table_or_coll_name, projection={"_id": False, "t": True})

        # has_in_operator = query.get("in", False) if query is not None else False

        postgres_query = SQL("SELECT {projection} FROM {table_name}").format(
            projection=SQL(", ").join([Identifier(projection) for projection in projection])
            if projection is not None
            else SQL("*"),
            table_name=Identifier(table_name),
        )

        if query is not None:
            # print(query["value"])
            has_in_operator = query.get("in", False)
            postgres_query += SQL(" WHERE {column} {equal_or_in} {value}").format(
                column=Identifier(query["attribute"]),
                equal_or_in=SQL("IN") if has_in_operator else SQL("="),
                # value=Literal(query["value"]) if has_in_operator else Literal(query["value"]),
                value=Literal(query["value"]) if not has_in_operator else Literal(tuple(query["value"])),
                # else SQL(", ").join(Literal(int(val)) for val in query["value"]),
            )

        # if has_in_operator:
        #     postgres_query = SQL("SELECT * WHERE {column} IN {values}").format(
        #         column=Identifier(query["attribute"]),
        #         values=SQL(", ").join([Identifier(val) for val in query["value"]]),
        #     )
        #     postgres_query += SQL(" ORDER BY {sort}").format(sort=SQL(sort))

        # else:
        #     postgres_query = SQL("SELECT {projection} FROM {table_name} WHERE {column} = {value}").format(
        #         projection=SQL(", ").join([Identifier(projection) for projection in projection])
        #         if projection is not None
        #         else SQL("*"),
        #         table_name=Identifier(table_name),
        #         column=Identifier(query["attribute"]) if query is not None else Identifier("*"),
        #         value=Literal(query["value"]) if query is not None else Literal("*"),
        #     )
        if sort:
            postgres_query += SQL(" ORDER BY {sort}").format(sort=SQL(sort))
        if limit:
            postgres_query += SQL(" LIMIT {limit}").format(limit=Literal(limit))
        if skip:
            postgres_query += SQL(" OFFSET {skip}").format(skip=Literal(skip))

        # print(postgres_query)
        self.execute(postgres_query)
        columns = [desc[0] for desc in self.dbms_db.description]
        result = [{columns[i]: row[i] for i in range(len(columns))} for row in self.dbms_db.fetchall()]
        # print(result)
        return list(result)

    def update_one(self, table_name, query, set_value, upsert=False) -> bool:
        # columns = payload_insert.keys()
        # values = tuple(payload_insert.values())
        # print("query['value']", query["value"])
        # print("set_value", set_value)
        postgres_update = SQL("UPDATE {table_name} SET {column} = {set_value} WHERE {column} = {value}").format(
            table_name=Identifier(table_name),
            column=Identifier(query["attribute"]),
            value=Literal(query["value"]),
            set_value=Literal(set_value),
        )
        self.execute(postgres_update)
        if self.dbms_db.rowcount > 0:
            return True
        else:
            # If upsert is True and no rows were updated, insert a new row
            if upsert:
                postgres_update = SQL("INSERT INTO {table_name} ({column}) VALUES ({set_value})").format(
                    table_name=Identifier(table_name),
                    column=Identifier(query["attribute"]),
                    set_value=Literal(set_value),
                )
                self.execute(postgres_update)
                if self.dbms_db.rowcount > 0:
                    return True
        return False

    def insert_one(self, table_name, payload_insert) -> bool:
        # if filter_nan:
        #     payload_insert = get_filtered_from_nan(payload_insert)
        if len(payload_insert) == 0:
            self.log.warning("No data to insert_one into DB!")
            return False
        columns = payload_insert.keys()
        values = tuple(payload_insert.values())
        query = SQL("INSERT INTO {table_name} ({columns}) VALUES ({values})").format(
            table_name=Identifier(table_name),
            columns=SQL(", ").join(map(Identifier, columns)),
            values=SQL(", ").join(map(Literal, values)),
        )
        return self.execute(query)

    def insert_many(self, table_name, payload_insert: list[dict]) -> bool:
        # if filter_nan:
        #     payload_insert = get_filtered_from_nan(payload_insert)
        if len(payload_insert) == 0:
            self.log.warning("No data to insert_many into DB!")
            return False
        columns = payload_insert[0].keys()
        values = [tuple(d.values()) for d in payload_insert]
        query = SQL("INSERT INTO {table_name} ({columns}) VALUES {values}").format(
            table_name=Identifier(table_name),
            columns=SQL(", ").join(map(Identifier, columns)),
            values=SQL(", ").join(map(Literal, values)),
        )
        return self.execute(query)
