""" backend/postgres.py

Module: tradeforce.backend.postgres
-----------------------------------

Provides a PostgreSQL database interface that allows for basic CRUD operations
as well as other PostgreSQL specific operations. It includes functions for creating tables,
inserting data, querying data, updating data, and deleting data. Additionally, this module
provides functionality for creating indexes and handling upserts.

Main BackendSQL methods:

    create_index: Create an index on the specified table with a given index_name
                    and optional unique constraint.

    query:        Perform a query on the specified table with given filters,
                    projection, sorting, limit, and skip options.

    update_one:   Update a single row in the specified table based on the given
                    query and set_value, with optional upsert.

    insert_one:   Insert a single row into the specified table
                    using the provided payload.

    insert_many:  Insert multiple rows into the specified table
                    using the provided payload and optional chunk_size.

    delete_one:   Delete a single row from the specified table
                    that matches the given query.

Some helper methods:
    _create_update_query: Create an SQL update query for the specified table
                            using the provided filter and update values.

    _insert:              Insert multiple rows into the specified table
                            using the provided columns and values.

"""
from __future__ import annotations
from time import sleep

from psycopg2 import OperationalError
import psycopg2.pool as pool
from psycopg2.sql import SQL, Composed, Identifier, Literal
from typing import TYPE_CHECKING, Generator

from tradeforce.backend import Backend
from tradeforce.backend.sql_tables import CreateTables


# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import Tradeforce
    from psycopg2.extensions import connection, cursor


def chunk_data(data: list[tuple], chunk_size: int) -> Generator:
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


class BackendSQL(Backend):
    """
    Extends the Backend class to provide an interface for handling SQL operations
    with a PostgreSQL backend. Offers methods for connecting to the database,
    executing queries, and performing CRUD operations on the database tables.

    Attributes:
        connected:       A flag indicating whether the backend is connected to the database.
        reconnect_count: A counter for the number of reconnection attempts.
        create_table:    An instance of the CreateTables class for creating tables
                            in the database.
    """

    def __init__(self, root: Tradeforce):
        """Initializes the BackendSQL object

        by calling the parent class's __init__ method, setting initial values
        for the connected and reconnect_count attributes, creating an instance
        of the CreateTables class, and establishing a connection to the database.

        Args:
            root (Tradeforce): The root instance of the Tradeforce application
                                providing access to the config and logging modules
                                or any other module.
        """
        super().__init__(root)
        self.connected = False
        self.reconnect_count = 0
        self.create_table = CreateTables(root, self)
        self._establish_connection(db_name=self.config.dbms_db)

    def _is_connected(self, db_name: str) -> bool:
        """Attempts to establish a connection to the specified PostgreSQL database.

        Construct a database URI using the provided `db_name`. Then try to create a
        connection pool with a minimum of 1 and a maximum of 10 connections.

        If the connection is successful, the set `self.connected` to True
        Otherwise, set `self.connected` to False.

        Params:
            db_name: The name of the PostgreSQL database to connect to.

        Returns:
            bool: True if the connection to the database is established successfully,
                    False otherwise.
        """
        self.dbms_uri = self.construct_uri(db_name)

        try:
            self.pool = pool.SimpleConnectionPool(minconn=1, maxconn=10, dsn=self.dbms_uri)
            self.connected = True

        except OperationalError:
            self.log.info("Failed to connect to Postgres database %s.", db_name)
            self.connected = False
        return self.connected

    def _register_cursor(self) -> cursor:
        """Register and return the cursor of the database connection.

        Retrieve a connection from the connection pool and create a cursor
        associated with that connection. It also sets the 'autocommit' property
        of the  connection to True, allowing the execution of individual SQL
        statements without the need for an explicit transaction.

        Returns:
            cursor: The registered cursor of the PostgreSQL database connection.

        Notes:
            The registered cursor can be accessed via self.dbms_db.
        """
        pool_connection: connection = self.pool.getconn()

        with pool_connection as dbms_client:
            dbms_client.autocommit = True
            dbms_db: cursor = dbms_client.cursor()

        return dbms_db

    def _check_new_tables(self) -> None:
        """Check for the existence of specific tables in the database
        -> and create them if they do not exist.

        Check for the existence of the history entity table (defined in the
        config file) and update the 'is_new_history_entity' attribute accordingly.

        Verify if the tables 'trader_status', 'open_orders', and 'closed_orders'
        exist in the database. If any of these tables are not present, call the
        corresponding creation methods from the CreateTables instance
        (self.create_table) to create the missing tables.
        """
        self.is_new_history_entity = self.is_new_entity(self.config.dbms_history_entity_name)

        if self.is_new_entity("trader_status"):
            self.create_table.trader_status()

        if self.is_new_entity("open_orders"):
            self.create_table.open_orders()

        if self.is_new_entity("closed_orders"):
            self.create_table.closed_orders()

    def _try_reconnect(self, db_name: str) -> None:
        """Attempt to reconnect to the specified database
        -> considering the maximum number of attempts and the delay between retries.

        Increase the 'reconnect_count' attribute for each attempt, and if the maximum number of
        attempts is reached, raise a SystemExit exception. Update the 'reconnect_max_attempts'
        attribute accordingly.

        If the 'reconnect_count' attribute exceeds 3, a cool-down period (reconnect_delay_sec)
        is applied before the next connection attempt.

        Params:
            db_name: The name of the database to reconnect to.

        Raises:
            SystemExit: If the maximum number of attempts is reached.
        """

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
        """Establish a connection to the specified database
        -> and ensure necessary tables exist.

        If no database name is provided, connect to the default database specified
        in the configuration. Once connected, if the database name matches the
        configuration's connect_db, check for the existence of the required
        database or creates it.

        If the database name matches the configuration's dbms_db,
        check and create necessary tables.

        If the connection fails, trigger the _try_reconnect() method
        to attempt reconnection.

        Params:
            db_name: The name of the database to establish a connection to. Defaults to None.
        """

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
        """Execute the given SQL query.

        Attempt to execute the provided SQL query using the database connection's cursor.

        Params:
            query (Composed): The SQL query to execute.

        Returns:
            bool: True if the query execution is successful, False otherwise.
        """
        try:
            self.dbms_db.execute(query)
            execute_ok = True
        except (TypeError, OperationalError):
            execute_ok = False
            self.log.error("SQL execute failed!", exc_info=True)
            self.log.error("SQL query: %s", query.as_string(self.dbms_db))
        return execute_ok

    def _has_executed(self) -> bool:
        """Determine if the last SQL query executed successfully.

        Returns:
            True if the last SQL query executed successfully, False otherwise.
        """
        return self.dbms_db.rowcount > 0

    def db_exists_or_create(self, db_name: str | None = None) -> None:
        """Check if the specified database exists and create it if not.

        First check if the given database exists in the pg_catalog.pg_database.
        If the check fails, raise a SystemExit error with a relevant message.

        If the database does not exist, create the database.
        If the database already exists, proceed with the existing database.

        Params:
            db_name: The name of the database to check and create if necessary.
                Defaults to the value specified in the configuration.
        """

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

    def is_new_entity(self, table_name: str) -> bool:
        """Determine if a given table is new
        -> by checking its existence in the database.

        Note:
            "Entity" is either a SQL table or a MongoDB collection.
            is_new_entity method is also implemented in the BackendMongoDB
            interface and there refers to a Collection.

        Params:
            table_name: The name of the table to check for existence in the database.

        Returns:
            bool: True if the table is new, False otherwise.
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

    # --------------------------------#
    # PostgreSQL specific operations #
    # --------------------------------#

    def create_index(self, table_name: str, index_name: str, unique: bool = False) -> None:
        """Create an index on the specified table
        -> with the given index_name and optional unique constraint.

        Constructs an SQL query to create an index on the specified table.
        If the 'unique' parameter is set to True, add a UNIQUE constraint to the index.

        Params:
            table_name: The name of the table on which to create the index.
            index_name: The name of the index to be created.
            unique:     Whether to enforce a unique constraint on the index. Defaults to False.
        """

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
        """Perform a query on the specified table
        -> with the given filters, projection, sorting, limit, and skip options.

        Construct an SQL query to retrieve data from the specified table based on the
        provided filter criteria: projection, sorting, limit, and skip options.

        Return a list of dictionaries, where each dictionary represents a row in the table
        with keys being the column names and values being the corresponding cell values.

        Params:
            table_name: The name of the table to query.

            query:      Filter criteria for the query, as a dictionary with keys
                            "attribute", "value", and an optional "in" operator.

            projection: The columns to return in the query result.

            sort:       A list of tuples specifying the columns and the sort direction
                            (1 for ascending, -1 for descending).

            limit:      The maximum number of rows to return.

            skip:       The number of rows to skip before starting to return rows.

        Returns:
            A list of dictionaries, where each dictionary represents a row in the table.
        """
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
        """Create an SQL update query for the specified table
        -> using the provided filter and update values.

        Construct an SQL update query to modify the data in the specified table
        based on the provided filter criteria and update values.

        Params:
            table_name: The name of the table to update.

            query:      Filter criteria for the query, as a dictionary with keys
                            'attribute' and 'value'.

            set_value:  The new value(s) to set for the specified attribute(s).
                        Can be a single value or a dictionary with keys being
                        column names and values being the corresponding new values.

        Returns:
            A Composed object representing the SQL update query.
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
        upsert: bool = False,
    ) -> bool:
        """Update a single row in the specified table
        -> based on the given query and set_value.

        Update a single row in the specified table based on the provided filter criteria
        and new values. If the row does not exist and the upsert flag is True, insert the
        row with the provided values.

        Params:
            table_name: The name of the table to update.

            query:      Filter criteria for the query, as a dictionary with keys
                            'attribute' and 'value'.
            set_value:  The new value(s) to set for the specified attribute(s). Can be a
                            single value or a dictionary with keys being column names and
                            values being the corresponding new values.

            upsert:     If True, insert the row with the provided values if it does not exist.

        Returns:
            True if the update or upsert operation was successful, False otherwise.
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
        """Insert multiple rows into the specified table
        -> using the provided columns and values.

        Constructs and execute an SQL INSERT query for the specified table,
        columns, and values.

        Params:
            table_name:  The name of the table to insert data into.
            columns:     A list of column names for which data is being inserted.
            values_list: A list of lists, where each inner list contains values
                            corresponding to the provided columns.

        Returns:
            True if the insert operation was successful, False otherwise.
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
        """Insert a single row into the specified table
        -> using the provided payload.

        Constructs and execute an SQL INSERT query for the specified table and payload.

        Params:
            table_name:     The name of the table to insert data into.
            payload_insert: A dictionary containing key-value pairs, where each key
                                represents a column name and each value represents
                                the corresponding data to be inserted.

        Returns:
            True if the insert operation was successful, False otherwise.
        """
        if not payload_insert:
            self.log.warning("No data to insert into DB! [table: %s]", table_name)
            return False

        columns = list(payload_insert.keys())
        values = [tuple(payload_insert.values())]

        return self._insert(table_name, columns, values)

    def insert_many(self, table_name: str, payload_insert: list[dict], chunk_size: int = 1000) -> bool:
        """Insert multiple rows into the specified table
        -> using the provided payload.

        Construct and execute an SQL INSERT query for each chunk of the specified
        table and payload.

        Params:
            table_name:     The name of the table to insert data into.

            payload_insert: A list of dictionaries, where each dictionary contains
                                key-value pairs representing column names and
                                corresponding data to be inserted.

            chunk_size:     The number of rows to insert in each SQL INSERT query. Defaults to 1000.

        Returns:
            True if all insert operations were successful, False otherwise.
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
        """Delete a single row from the specified table
        -> that matches the given query.

        Construct and execute an SQL DELETE query for the specified
        table and query.

        Params:
            table_name: The name of the table to delete a row from.
            query:      A dictionary containing key-value pairs representing
                            the attribute (column name) and value to be
                            matched for the row to be deleted.

        Returns:
            True if the delete operation was successful, False otherwise.
        """
        sql_delete = SQL("DELETE FROM {table_name} WHERE {column} = {value}").format(
            table_name=Identifier(table_name),
            column=Identifier(query["attribute"]),
            value=Literal(query["value"]),
        )
        self.execute(sql_delete)
        return self._has_executed()
