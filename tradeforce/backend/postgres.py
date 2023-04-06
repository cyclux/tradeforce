""" tradeforce/backend/postgres.py

This module contains the BackendSQL class which is used to handle SQL operations with PostgreSQL backend.
"""
from __future__ import annotations
from time import sleep

from psycopg2 import OperationalError
import psycopg2.pool as pool
from psycopg2.sql import SQL, Composed, Identifier, Literal
from typing import TYPE_CHECKING

from tradeforce.backend import Backend
from tradeforce.backend.sql_tables import CreateTables


# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import Tradeforce
    from psycopg2.extensions import connection, cursor


def chunk_data(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


class BackendSQL(Backend):
    """Interface to handle SQL operations with PostgreSQL backend."""

    def __init__(self, root: Tradeforce):
        """Initializes the BackendSQL object and establishes a connection to the database."""

        super().__init__(root)
        self.connected = False
        self.reconnect_count = 0
        self.create_table = CreateTables(root, self)
        self._establish_connection(db_name=self.config.dbms_db)

    def _is_connected(self, db_name: str) -> bool:
        """Connects to the database with the given DB name."""

        self.dbms_uri = self.construct_uri(db_name)
        try:
            self.pool = pool.SimpleConnectionPool(minconn=1, maxconn=10, dsn=self.dbms_uri)
            self.connected = True

        except OperationalError:
            self.log.info("Failed to connect to Postgres database %s.", db_name)
            self.connected = False
        return self.connected

    def _register_cursor(self) -> cursor:
        """Registers the cursor of the database connection.

        Can be accessed via self.dbms_db.
        """
        pool_connection: connection = self.pool.getconn()
        with pool_connection as dbms_client:
            dbms_client.autocommit = True
            dbms_db: cursor = dbms_client.cursor()
        return dbms_db

    def _check_new_tables(self) -> None:
        self.is_new_history_entity = self.is_new_entity(self.config.dbms_history_entity_name)
        if self.is_new_entity("trader_status"):
            self.create_table.trader_status()
        if self.is_new_entity("open_orders"):
            self.create_table.open_orders()
        if self.is_new_entity("closed_orders"):
            self.create_table.closed_orders()

    def _try_reconnect(self, db_name: str):
        """Tries to reconnect to the database."""

        self.reconnect_count += 1

        if self.reconnect_max_attempts == 0:
            raise SystemExit(f"[ERROR] Max tries reached to connect to {db_name}.")

        if self.reconnect_max_attempts != 0:
            if self.reconnect_max_attempts > 0:
                self.reconnect_max_attempts -= 1

            is_cool_down = self.reconnect_count > 3

            self.log.info(
                "Retrying connection to %s in %s seconds...",
                db_name,
                self.reconnect_delay_sec if is_cool_down else 0,
            )
            if is_cool_down:
                sleep(self.reconnect_delay_sec)

            self._establish_connection(db_name)

    def _establish_connection(self, db_name: str | None = None) -> None:
        """Establishes a connection to the database by checking and creating necessary tables."""

        if db_name is None:
            db_name = self.config.dbms_connect_db

        if self._is_connected(db_name):
            self.log.info("Connected to database %s.", db_name)
            self.dbms_db = self._register_cursor()

            if db_name == self.config.dbms_connect_db:
                self.db_exists_or_create()
                self.dbms_db.close()
                self._establish_connection(self.config.dbms_db)

            if db_name == self.config.dbms_db:
                self._check_new_tables()
        else:
            self._try_reconnect(self.config.dbms_connect_db)

    def execute(self, query: Composed) -> bool:
        """Executes the provided SQL query and returns a boolean indicating success or failure."""

        try:
            self.dbms_db.execute(query)
            execute_ok = True
        except (TypeError, OperationalError):
            execute_ok = False
            self.log.error("SQL execute failed!", exc_info=True)
            self.log.error("SQL query: %s", query.as_string(self.dbms_db))
        return execute_ok

    def _has_executed(self):
        return self.dbms_db.rowcount > 0

    def db_exists_or_create(self, db_name: str | None = None):
        """Checks if the database exists and creates it if not."""

        db_name = db_name or self.config.dbms_db

        query = SQL("SELECT 1 FROM pg_catalog.pg_database WHERE datname = {db_name};").format(db_name=Literal(db_name))
        execute_ok = self.execute(query)
        if not execute_ok:
            raise SystemExit(
                f"[ERROR] Failed to check pg_catalog.pg_database of {db_name}. " + "Choose different dbms_connect_db."
            )
        db_exists = self.dbms_db.fetchone()
        if not db_exists:
            query = SQL("CREATE DATABASE {db_name};").format(db_name=Identifier(db_name))
            self.execute(query)
            self.log.info("Created database %s", db_name)
        else:
            self.log.info("Database already exists. Switching to %s ...", db_name)

    def is_new_entity(self, table_name: str):
        """Checks if the table exists and returns a boolean indicating the result.
        Note: "Entity" is either a SQL table or a MongoDB collection.
        is_new_entity method is also implemented in the BackendMongoDB interface.

        """

        query = SQL("SELECT 1 FROM pg_catalog.pg_tables WHERE tablename = {table_name};").format(
            table_name=Literal(table_name)
        )
        self.execute(query)
        table_exists = self.dbms_db.fetchone()
        if not table_exists:
            return True
        else:
            return False

    ##################################
    # PostgreSQL specific operations #
    ##################################

    def create_index(self, table_name: str, index_name: str, unique=False) -> None:
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

    def _create_update_query(
        self,
        table_name: str,
        query: dict[str, str | int | list],
        set_value: str | int | list | dict,
    ) -> Composed:
        """Helper method which creates an update query for the specified table
        and returns it as a Composed object.
        """

        if isinstance(set_value, dict):
            columns = set_value.keys()
            values = tuple(set_value.values())
            set_expr = SQL(", ").join(
                SQL("{} = {}").format(Identifier(col), Literal(val)) for col, val in zip(columns, values)
            )
        else:
            set_expr = SQL("{} = {}").format(Identifier(query["attribute"]), Literal(set_value))

        return SQL("UPDATE {table_name} SET {set_expr} WHERE {column} = {value}").format(
            table_name=Identifier(table_name),
            set_expr=set_expr,
            column=Identifier(query["attribute"]),
            value=Literal(query["value"]),
        )

    def update_one(
        self,
        table_name: str,
        query: dict[str, str | int | list],
        set_value: str | int | list | dict,
        upsert=False,
    ) -> bool:
        """Updates a single record in the specified table
        and returns a boolean indicating success or failure.
        """

        postgres_update = self._create_update_query(table_name, query, set_value)
        self.execute(postgres_update)
        update_executed = self._has_executed()
        if update_executed:
            return True
        if upsert and isinstance(set_value, dict) and query["attribute"] == "t":
            set_value["t"] = query["value"]
            insert_executed = self.insert_one(table_name, set_value)
            if insert_executed:
                return True
        return False

    def _insert(self, table_name: str, columns: list, values_list: list) -> bool:
        """Helper method for inserting data into DB.
        Returns a boolean indicating success or failure.
        """

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
        return self._has_executed()

    def insert_one(self, table_name: str, payload_insert: dict) -> bool:
        """Inserts a single record into the specified table
        and returns a boolean indicating success or failure.
        """

        if not payload_insert:
            self.log.warning("No data to insert into DB! [table: %s]", table_name)
            return False
        columns = list(payload_insert.keys())
        values = [tuple(payload_insert.values())]
        return self._insert(table_name, columns, values)

    def insert_many(self, table_name: str, payload_insert: list[dict], chunk_size: int = 1000) -> bool:
        """Inserts multiple records into the specified table in chunks
        and returns a boolean indicating success or failure.
        """
        if not payload_insert or len(payload_insert) < 1:
            self.log.warning("No data to insert into DB! [table: %s]", table_name)
            return False

        columns = list(payload_insert[0].keys())
        values_list = [tuple(d.values()) for d in payload_insert]

        success = True
        for chunk in chunk_data(values_list, chunk_size):
            success &= self._insert(table_name, columns, chunk)

        return success

    def delete_one(self, table_name: str, query: dict[str, str | int]) -> bool:
        """Deletes a single record from the specified table
        and returns a boolean indicating success or failure.
        """

        sql_delete = SQL("DELETE FROM {table_name} WHERE {column} = {value}").format(
            table_name=Identifier(table_name),
            column=Identifier(query["attribute"]),
            value=Literal(query["value"]),
        )
        self.execute(sql_delete)
        return self._has_executed()
