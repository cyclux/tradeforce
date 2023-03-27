""" tradeforce/backend/postgres.py

This module contains the BackendSQL class which is used to handle SQL operations with PostgreSQL backend.
"""
from __future__ import annotations
import sys

from psycopg2.extensions import connection, cursor
from psycopg2 import OperationalError
import psycopg2.pool as pool
from psycopg2.sql import SQL, Composed, Identifier, Literal
from typing import TYPE_CHECKING

from tradeforce.backend import Backend
from tradeforce.backend.sql_tables import CreateTables


# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import TradingEngine


class BackendSQL(Backend):
    """Interface to handle SQL operations with PostgreSQL backend."""

    def __init__(self, root: TradingEngine):
        """Initializes the BackendSQL object and establishes a connection to the database."""

        super().__init__(root)
        self.connected = False
        self.create_table = CreateTables(root, self)
        self.establish_connection(db_name=self.config.dbms_db)
        # Only sync state with backend now if there is no exchange API connection.
        # So in case config.run_live is True (API connection present)
        # both db_sync_state_trader() and db_sync_state_orders()
        # will be called by ExchangeWebsocket.ws_priv_wallet_snapshot()
        if self.config.use_dbms and not self.config.run_live:
            self.db_sync_state_trader()
            self.db_sync_state_orders()

    def connect(self, db_name) -> None:
        """Connects to the database with given DB name."""

        dbms_uri = self.construct_uri(db_name)
        try:
            self.pool = pool.SimpleConnectionPool(minconn=1, maxconn=10, dsn=dbms_uri)
        except OperationalError:
            self.log.info("Failed to connect to database %s. Trying again...", db_name)
            self.connected = False
            pass
        else:
            self.connected = True

    def establish_connection(self, db_name=None) -> None:
        """Establishes a connection to the database by checking and creating necessary tables."""

        self.connect(db_name)
        if self.connected:
            pool_connection: connection = self.pool.getconn()
            with pool_connection as dbms_client:
                dbms_client.autocommit = True
                self.dbms_db: cursor = dbms_client.cursor()

            if db_name == self.config.dbms_connect_db:
                self.db_exists_or_create()
                self.dbms_db.close()
                self.establish_connection(self.config.dbms_db)
            if db_name == self.config.dbms_db:
                self.is_new_coll_or_table = self.check_table()
                if self.is_new_coll_or_table:
                    self.create_table.trader_status()
                    self.create_table.open_orders()
                    self.create_table.closed_orders()

        else:
            self.establish_connection(self.config.dbms_connect_db)

    def execute(self, query: Composed) -> bool:
        """Executes the provided SQL query and returns a boolean indicating success or failure."""

        try:
            self.dbms_db.execute(query)
            execute_ok = True
        except (TypeError, OperationalError):
            execute_ok = False
            self.log.error("SQL execute failed!")
        return execute_ok

    def db_exists_or_create(self):
        """Checks if the database exists and creates it if not."""

        query = SQL("SELECT 1 FROM pg_catalog.pg_database WHERE datname = {db_name};").format(
            db_name=Literal(self.config.dbms_db)
        )
        execute_ok = self.execute(query)
        if not execute_ok:
            sys.exit(
                f"[ERROR] Failed to check pg_catalog.pg_database of {self.config.dbms_db}. "
                + "Choose different dbms_connect_db."
            )
        db_exists = self.dbms_db.fetchone()
        if db_exists is None:
            query = SQL("CREATE DATABASE {db_name};").format(db_name=Identifier(self.config.dbms_db))
            self.execute(query)
            self.log.info("Created database %s", self.config.dbms_db)
        else:
            self.log.info("Database %s already exists", self.config.dbms_db)

    def check_table(self):
        """Checks if the table exists and returns a boolean indicating the result."""

        query = SQL("SELECT 1 FROM pg_catalog.pg_tables WHERE tablename = {table_name};").format(
            table_name=Literal(self.config.dbms_entity_name)
        )
        self.execute(query)
        table_exists = self.dbms_db.fetchone()
        if table_exists is None:
            is_new_table = True
        else:
            is_new_table = False
            if self.config.force_source != "postgresql":
                if self.config.force_source == "api":
                    sys.exit(
                        f"[ERROR] PostgreSQL history '{self.config.dbms_entity_name}' already exists. "
                        + "Cannot load history via API. Choose different history DB name or loading method."
                    )
                self.sync_check_needed = True
        return is_new_table

    ##################################
    # PostgreSQL specific operations #
    ##################################

    def create_index(self, table_name, index_name, unique=False) -> None:
        """Creates an index on the specified table with the given index_name and optional unique constraint."""

        query = SQL("CREATE {unique} INDEX {index_name} ON {table_name} ({index_name});").format(
            unique=SQL("UNIQUE") if unique else SQL(""),
            index_name=Identifier(index_name),
            table_name=Identifier(table_name),
        )
        self.execute(query)
        self.log.info("Created index %s on %s", index_name, table_name)

    def query(
        self,
        table_name: str,
        query: dict[str, str | int | list] | None = None,
        projection: dict[str, bool] | None = None,
        sort: list[tuple[str, int]] | None = None,
        limit: int | None = None,
        skip: int | None = None,
    ) -> list:
        """Queries the database with the provided parameters and returns the result as a list of dictionaries."""

        postgres_query = SQL("SELECT {projection} FROM {table_name}").format(
            projection=SQL(", ").join([Identifier(projection) for projection in projection])
            if projection is not None
            else SQL("*"),
            table_name=Identifier(table_name),
        )

        if query is not None:
            has_in_operator = query.get("in", False)
            query_value = tuple(query["value"]) if isinstance(query["value"], list) else query["value"]
            postgres_query += SQL(" WHERE {column} {equal_or_in} {value}").format(
                column=Identifier(query["attribute"]),
                equal_or_in=SQL("IN") if has_in_operator else SQL("="),
                value=Literal(query["value"]) if not has_in_operator else Literal(query_value),
            )

        if sort is not None:
            postgres_query += SQL(" ORDER BY {sort}").format(sort=SQL(sort[0][0]))
            postgres_query += SQL(" {direction}").format(direction=SQL("ASC") if sort[0][1] == 1 else SQL("DESC"))

        if limit is not None:
            postgres_query += SQL(" LIMIT {limit}").format(limit=Literal(limit))

        if skip is not None:
            postgres_query += SQL(" OFFSET {skip}").format(skip=Literal(skip))

        self.execute(postgres_query)
        columns = [desc[0] for desc in self.dbms_db.description]
        result = [{columns[i]: row[i] for i in range(len(columns))} for row in self.dbms_db.fetchall()]
        return list(result)

    def update_one(
        self, table_name: str, query: dict[str, str | int | list], set_value: str | int | list | dict, upsert=False
    ) -> bool:
        """Updates a single record in the specified table and returns a boolean indicating success or failure."""

        set_val_is_dict = isinstance(set_value, dict)
        if isinstance(set_value, dict):
            columns = set_value.keys()
            values = tuple(set_value.values())
        postgres_update = SQL("UPDATE {table_name} SET ({columns}) = ({set_value}) WHERE {column} = {value}").format(
            table_name=Identifier(table_name),
            columns=SQL(", ").join(map(Identifier, columns)) if set_val_is_dict else Identifier(columns),
            column=Identifier(query["attribute"]),
            value=Literal(query["value"]),
            set_value=SQL(", ").join(map(Literal, values)) if set_val_is_dict else Literal(set_value),
        )
        self.execute(postgres_update)
        if self.dbms_db.rowcount > 0:
            return True
        else:
            if upsert:
                if isinstance(set_value, dict):
                    if query["attribute"] == "t":
                        set_value["t"] = query["value"]
                    self.insert_one(table_name, set_value)
                if self.dbms_db.rowcount > 0:
                    return True
        return False

    def _insert(self, table_name: str, columns: list, values_list: list) -> bool:
        """A helper method for inserting data into DB. Returns a boolean indicating success or failure"""

        if not values_list:
            self.log.warning("No data to insert into DB!")
            return False

        sql_insert = SQL("INSERT INTO {table_name} ({columns}) VALUES {values_list}").format(
            table_name=Identifier(table_name),
            columns=SQL(", ").join(map(Identifier, columns)),
            values_list=SQL(", ").join(
                SQL("({values})").format(values=SQL(", ").join(map(Literal, values))) for values in values_list
            ),
        )
        self.execute(sql_insert)
        return self.dbms_db.rowcount > 0

    def insert_one(self, table_name: str, payload_insert: dict) -> bool:
        """Inserts a single record into the specified table and returns a boolean indicating success or failure."""

        columns = list(payload_insert.keys())
        values = [tuple(payload_insert.values())]
        return self._insert(table_name, columns, values)

    def insert_many(self, table_name: str, payload_insert: list[dict]) -> bool:
        """Inserts multiple records into the specified table and returns a boolean indicating success or failure."""

        columns = list(payload_insert[0].keys())
        values_list = [tuple(d.values()) for d in payload_insert]
        return self._insert(table_name, columns, values_list)

    def delete_one(self, table_name: str, query: dict[str, str | int]) -> bool:
        """Deletes a single record from the specified table and returns a boolean indicating success or failure."""

        sql_delete = SQL("DELETE FROM {table_name} WHERE {column} = {value}").format(
            table_name=Identifier(table_name),
            column=Identifier(query["attribute"]),
            value=Literal(query["value"]),
        )
        self.execute(sql_delete)
        return self.dbms_db.rowcount > 0
