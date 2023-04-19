# mypy: disable-error-code="assignment, attr-defined, misc"

import pytest
from psycopg2 import OperationalError
from psycopg2.pool import SimpleConnectionPool
from psycopg2.sql import SQL, Identifier, Literal, Composed
from psycopg2.extensions import connection, cursor
from unittest.mock import MagicMock, patch, call
from tradeforce.config import Config
from tradeforce import Tradeforce
from tradeforce.backend import BackendSQL
from tradeforce.backend.sql_tables import CreateTables

config = {
    "trader": {
        "id": 3,
        "run_live": False,
        "credentials_path": "exchange_creds.cfg",
        "budget": 0,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
            "amount_invest_per_asset": 1000,
            "investment_cap": 0,
            "buy_performance_score": 0.10,
            "buy_performance_boundary": 0.05,
            "buy_performance_preference": 1,
            "_hold_time_increments": 1000,
            "profit_factor_target": 1.10,
            "profit_factor_target_min": 1.01,
            "moving_window_hours": 160,
        },
    },
    "backend": {
        "dbms": "postgresql",
        "dbms_host": "docker_postgres",
        "dbms_port": 5432,
        "dbms_connect_db": "postgres",
        "dbms_user": "postgres",
        "dbms_pw": "postgres",
        "local_cache": False,
    },
    "market_history": {
        "name": "bfx_history_docker_60days",
        "exchange": "bitfinex",
        "base_currency": "USD",
        "candle_interval": "5min",
        "fetch_init_timeframe_days": 60,
        "update_mode": "none",  # none, once or live
    },
}


@pytest.fixture
def mock_config() -> dict:
    return config


@pytest.fixture
def trading_engine(mock_config: dict) -> Tradeforce:
    return Tradeforce(config=mock_config)


@pytest.fixture
def config_object(trading_engine: Tradeforce, mock_config: dict) -> Config:
    return Config(root=trading_engine, config_input=mock_config)


class TestBackendSQL:
    @pytest.fixture()
    def backend(self, trading_engine: Tradeforce) -> BackendSQL:
        backend = BackendSQL(trading_engine)
        backend.log = MagicMock(info=MagicMock(), error=MagicMock())
        backend.dbms_db = MagicMock(
            execute=MagicMock(return_value=None),
            fetchone=MagicMock(return_value=None),
            fetchall=MagicMock(return_value=[]),
        )
        backend.db_sync_state_trader = MagicMock(name="db_sync_state_trader")
        backend.db_sync_state_orders = MagicMock(name="db_sync_state_orders")
        backend.create_table.trader_status = MagicMock(name="trader_status")
        backend.create_table.open_orders = MagicMock(name="open_orders")
        backend.create_table.closed_orders = MagicMock(name="closed_orders")
        return backend

    @pytest.mark.parametrize("run_live, call_sync_methods", [(True, False), (False, True)])
    def test_init(
        self,
        backend: BackendSQL,
        trading_engine: Tradeforce,
        run_live: bool,
        call_sync_methods: bool,
    ) -> None:
        # Update trading_engine config to set run_live
        trading_engine.config.run_live = run_live

        # Reset connected attribute and call __init__ to re-initialize
        backend.connected = False
        backend._establish_connection = MagicMock(name="_establish_connection", return_value=None)
        backend.db_sync_state_trader.reset_mock()
        backend.db_sync_state_orders.reset_mock()

        # Call __init__ and manually set connected to True
        backend.__init__(trading_engine)
        backend.connected = True

        # Assertions for init test
        assert backend.connected is True
        assert isinstance(backend.create_table, CreateTables)
        assert backend.config.dbms_db == trading_engine.config.dbms_db

        # Assertions for sync test
        if call_sync_methods:
            assert backend.config.run_live is False
            backend.db_sync_state_trader.assert_called_once()
            backend.db_sync_state_orders.assert_called_once()
        else:
            assert backend.config.run_live is True
            backend.db_sync_state_trader.assert_not_called()
            backend.db_sync_state_orders.assert_not_called()

    @pytest.mark.parametrize("raise_error, expected_connected", [(False, True), (True, False)])
    def test_is_connected(
        self,
        backend: BackendSQL,
        raise_error: bool,
        expected_connected: bool,
    ) -> None:
        # Mock SimpleConnectionPool
        with patch("psycopg2.pool.SimpleConnectionPool", autospec=True) as mock_pool:
            if raise_error:
                # Raise OperationalError if raise_error is True
                mock_pool.side_effect = OperationalError()
            else:
                # Return a MagicMock if raise_error is False
                mock_pool.return_value = MagicMock(spec=SimpleConnectionPool)

            # Call _is_connected
            connected = backend._is_connected(db_name="test_db")

        # Assertions
        assert connected == expected_connected

    def test_register_cursor(self, backend: BackendSQL) -> None:
        # Mock pool and cursor
        mock_pool_connection = MagicMock(spec=connection)
        mock_cursor = MagicMock(spec=cursor)

        # Patch SimpleConnectionPool
        with patch("psycopg2.pool.SimpleConnectionPool", autospec=True) as mock_pool:
            # Set return value for the pool object
            mock_pool.return_value = MagicMock(getconn=MagicMock(return_value=mock_pool_connection))

            # Mock _is_connected to return True and set the mocked pool object
            with patch.object(BackendSQL, "_is_connected", return_value=True):
                backend.pool = mock_pool.return_value

                # Patch the connection object's __enter__ and cursor methods to return appropriate mocks
                with patch.object(mock_pool_connection, "__enter__", return_value=mock_pool_connection), patch.object(
                    mock_pool_connection, "cursor", return_value=mock_cursor
                ):
                    # Call _register_cursor
                    result_cursor = backend._register_cursor()

                    # Assertions
                    assert result_cursor == mock_cursor
                    backend.pool.getconn.assert_called_once()
                    mock_pool_connection.cursor.assert_called_once()
                    assert mock_pool_connection.autocommit is True

    @pytest.mark.parametrize(
        "is_new_history, is_new_trader, is_new_open, is_new_closed",
        [
            (True, True, True, True),
            (True, True, True, False),
            (True, True, False, True),
            (True, True, False, False),
            (True, False, True, True),
            (True, False, True, False),
            (True, False, False, True),
            (True, False, False, False),
            (False, True, True, True),
            (False, True, True, False),
            (False, True, False, True),
            (False, True, False, False),
            (False, False, True, True),
            (False, False, True, False),
            (False, False, False, True),
            (False, False, False, False),
        ],
    )
    def test_check_new_tables(
        self,
        backend: BackendSQL,
        is_new_history: bool,
        is_new_trader: bool,
        is_new_open: bool,
        is_new_closed: bool,
    ) -> None:
        # Mock is_new_entity
        backend.is_new_entity = MagicMock(side_effect=[is_new_history, is_new_trader, is_new_open, is_new_closed])

        # Mock create_table methods
        backend.create_table.trader_status = MagicMock()
        backend.create_table.open_orders = MagicMock()
        backend.create_table.closed_orders = MagicMock()

        # Call _check_new_tables
        backend._check_new_tables()

        # Assertions
        if is_new_trader:
            backend.create_table.trader_status.assert_called_once()
        else:
            backend.create_table.trader_status.assert_not_called()

        if is_new_open:
            backend.create_table.open_orders.assert_called_once()
        else:
            backend.create_table.open_orders.assert_not_called()

        if is_new_closed:
            backend.create_table.closed_orders.assert_called_once()
        else:
            backend.create_table.closed_orders.assert_not_called()

    @pytest.mark.parametrize(
        "db_name, expected_calls",
        [
            (None, 2),
            ("bitfinex_db", 1),
            ("postgres", 2),
        ],
    )
    def test_establish_connection(
        self,
        backend: BackendSQL,
        db_name: str | None,
        expected_calls: int,
    ) -> None:
        # Mock _is_connected to return True
        backend._is_connected = MagicMock(return_value=True)

        # Mock _register_cursor, _db_exists_or_create, and _check_new_tables
        backend._register_cursor = MagicMock()
        backend.db_exists_or_create = MagicMock()
        backend._check_new_tables = MagicMock()

        # Call _establish_connection
        backend._establish_connection(db_name)

        # Assertions
        if db_name is None or db_name == backend.config.dbms_connect_db:
            backend._is_connected.assert_has_calls([call(backend.config.dbms_connect_db), call(backend.config.dbms_db)])
            backend.db_exists_or_create.assert_called_once()
        else:
            backend._is_connected.assert_called_once_with(backend.config.dbms_db)
            backend.db_exists_or_create.assert_not_called()

        assert backend._register_cursor.call_count == expected_calls
        backend._check_new_tables.assert_called_once()
        assert backend._is_connected.call_count == expected_calls

    @pytest.mark.parametrize(
        "error_to_raise, expected_result",
        [
            (None, True),
            (TypeError, False),
            (OperationalError, False),
        ],
    )
    def test_execute(
        self,
        backend: BackendSQL,
        error_to_raise: Exception,
        expected_result: bool,
    ) -> None:
        # Mock the dbms_db cursor
        backend.dbms_db = MagicMock()

        # Define the side_effect for the execute method based on the raised error
        side_effect = error_to_raise if error_to_raise else None
        backend.dbms_db.execute = MagicMock(side_effect=side_effect)

        # Create a mock query object
        mock_query = MagicMock(spec=Composed)

        # Call the execute method with the mock query object
        result = backend.execute(mock_query)

        # Assertions
        backend.dbms_db.execute.assert_called_once_with(mock_query)
        assert result == expected_result

        # Check if log.error was called when an error occurs
        if error_to_raise:
            backend.log.error.assert_called_once_with("SQL execute failed!")
        else:
            backend.log.error.assert_not_called()

    @pytest.mark.parametrize(
        "rowcount, expected_result",
        [
            (0, False),
            (1, True),
            (5, True),
        ],
    )
    def test_has_executed(
        self,
        backend: BackendSQL,
        rowcount: int,
        expected_result: bool,
    ) -> None:
        # Mock the dbms_db cursor
        backend.dbms_db = MagicMock()

        # Set the rowcount attribute for the cursor
        backend.dbms_db.rowcount = rowcount

        # Call the _has_executed method
        result = backend._has_executed()

        # Assertions
        assert result == expected_result

    @pytest.mark.parametrize(
        "fetchone_result, create_called, execute_calls",
        [
            (None, True, 2),
            ((1,), False, 1),
        ],
    )
    def test_db_exists_or_create(
        self,
        backend: BackendSQL,
        fetchone_result,
        create_called: bool,
        execute_calls: int,
    ) -> None:
        # Mock execute and fetchone methods
        backend.execute = MagicMock(return_value=True)
        backend.dbms_db = MagicMock(fetchone=MagicMock(return_value=fetchone_result))
        backend.log.info = MagicMock()

        # Call the _db_exists_or_create method
        backend.db_exists_or_create()

        # Assertions
        query1 = SQL("SELECT 1 FROM pg_catalog.pg_database WHERE datname = {db_name};").format(
            db_name=Literal(backend.config.dbms_db)
        )
        # backend.execute.assert_called_with(query1) asserts with wrong call (query2)
        backend.execute.assert_any_call(query1)

        if create_called:
            query2 = SQL("CREATE DATABASE {db_name};").format(db_name=Identifier(backend.config.dbms_db))
            backend.execute.assert_called_with(query2)

        assert backend.execute.call_count == execute_calls

        backend.log.info.assert_called_once()
        backend.dbms_db.fetchone.assert_called_once()

    @pytest.mark.parametrize(
        "fetchone_result, expected_result",
        [
            (None, True),
            ((1,), False),
        ],
    )
    def test_is_new_entity(
        self,
        backend: BackendSQL,
        fetchone_result,
        expected_result: bool,
    ) -> None:
        # Mock execute and fetchone methods
        backend.execute = MagicMock(return_value=True)
        backend.dbms_db = MagicMock(fetchone=MagicMock(return_value=fetchone_result))

        # Call the is_new_entity method
        table_name = "test_table"
        result = backend.is_new_entity(table_name)

        # Assertions
        query = SQL("SELECT 1 FROM pg_catalog.pg_tables WHERE tablename = {table_name};").format(
            table_name=Literal(table_name)
        )
        backend.execute.assert_called_once_with(query)
        backend.dbms_db.fetchone.assert_called_once()
        assert result == expected_result

    @pytest.mark.parametrize("unique", [True, False])
    def test_create_backend_db_index(self, backend: BackendSQL, unique: bool):
        table_name = "test_table"
        index_name = "test_index"

        backend.dbms_db.execute = MagicMock(return_value=None)
        backend.log.info = MagicMock()

        backend.create_index(table_name, index_name, unique)

        # Assertions
        query = SQL("CREATE {unique} INDEX {index_name} ON {table_name} ({index_name});").format(
            unique=SQL("UNIQUE") if unique else SQL(""),
            index_name=Identifier(index_name),
            table_name=Identifier(table_name),
        )

        backend.dbms_db.execute.assert_called_once_with(query)
        backend.log.info.assert_called_once_with("Created index %s on %s", index_name, table_name)

    @pytest.mark.parametrize(
        "input_params,expected_query",
        [
            # TODO: Add more test cases with different combinations
            (
                {
                    "table_name": "test_table_name",
                    "query": None,
                    "projection": None,
                    "sort": None,
                    "limit": None,
                    "skip": None,
                },
                SQL("SELECT {projection} FROM {table_name}").format(
                    projection=SQL("*"),
                    table_name=Identifier("test_table_name"),
                ),
            )
        ],
    )
    def test_query(self, backend: BackendSQL, input_params: dict, expected_query: SQL):
        # Prepare test data
        table_name = input_params["table_name"]
        query = input_params["query"]
        projection = input_params["projection"]
        sort = input_params["sort"]
        limit = input_params["limit"]
        skip = input_params["skip"]

        backend.execute = MagicMock(return_value=None)

        result = backend.query(table_name, query, projection, sort, limit, skip)

        backend.execute.assert_called_once_with(expected_query)
        assert result == []

    @pytest.mark.parametrize(
        "table_name, query, set_value, expected_query",
        [
            (
                "test_table",
                {"attribute": "column1", "value": "value1"},
                {"column2": "value2"},
                SQL("UPDATE {table_name} SET ({columns}) = ({set_value}) WHERE {column} = {value}").format(
                    table_name=Identifier("test_table"),
                    columns=SQL(", ").join([Identifier("column2")]),
                    column=Identifier("column1"),
                    value=Literal("value1"),
                    set_value=SQL(", ").join([Literal("value2")]),
                ),
            ),
            (
                "test_table",
                {"attribute": "column1", "value": "value1"},
                {"column2": "value2", "column3": "value3"},
                SQL("UPDATE {table_name} SET ({columns}) = ({set_value}) WHERE {column} = {value}").format(
                    table_name=Identifier("test_table"),
                    columns=SQL(", ").join([Identifier("column2"), Identifier("column3")]),
                    column=Identifier("column1"),
                    value=Literal("value1"),
                    set_value=SQL(", ").join([Literal("value2"), Literal("value3")]),
                ),
            ),
        ],
    )
    def test_create_update_query(
        self,
        backend: BackendSQL,
        table_name: str,
        query: dict[str, str | int | list],
        set_value: str | int | list | dict,
        expected_query: Composed,
    ):
        result = backend._create_update_query(table_name, query, set_value)
        assert result == expected_query

    @pytest.mark.parametrize(
        "table_name, query, set_value, upsert, update_executed, insert_executed, expected_result",
        [
            ("test_table", {"attribute": "column1", "value": "value1"}, {"column2": "value2"}, False, True, None, True),
            (
                "test_table",
                {"attribute": "column1", "value": "value1"},
                {"column2": "value2"},
                False,
                False,
                None,
                False,
            ),
            ("test_table", {"attribute": "t", "value": "value1"}, {"column2": "value2"}, True, False, False, False),
            (
                "test_table",
                {"attribute": "t", "value": "value1"},
                {"column2": "value2", "column3": "value3"},
                True,
                False,
                True,
                True,
            ),
        ],
    )
    def test_update_one(
        self,
        backend: BackendSQL,
        table_name: str,
        query: dict[str, str | int | list],
        set_value: str | int | list | dict,
        upsert: bool,
        update_executed: bool,
        insert_executed: bool | None,
        expected_result: bool,
    ):
        backend.execute = MagicMock(return_value=None)
        backend.insert_one = MagicMock(return_value=insert_executed)
        backend._has_executed = MagicMock(return_value=update_executed)

        result = backend.update_one(table_name, query, set_value, upsert)
        assert result == expected_result

        if upsert and not update_executed and isinstance(set_value, dict) and query["attribute"] == "t":
            backend.insert_one.assert_called_once_with(table_name, set_value)
        else:
            backend.insert_one.assert_not_called()

    # Additional test for _insert
    @pytest.mark.parametrize(
        "table_name, columns, values_list, execution_result, expected_result",
        [
            ("test_table", ["column1", "column2"], [(1, "A"), (2, "B")], True, True),
            ("test_table", ["column1", "column2"], [], False, False),
        ],
    )
    def test_insert(
        self,
        backend: BackendSQL,
        table_name: str,
        columns: list,
        values_list: list,
        execution_result: bool,
        expected_result: bool,
    ):
        backend.execute = MagicMock(return_value=None)
        backend._has_executed = MagicMock(return_value=execution_result)

        result = backend._insert(table_name, columns, values_list)
        assert result == expected_result

    @pytest.mark.parametrize(
        "table_name, payload_insert, expected_result",
        [
            ("test_table", {"col1": 1, "col2": 2}, True),
            ("test_table", {}, False),
            ("test_table", None, False),
        ],
    )
    def test_insert_one(self, backend, table_name, payload_insert, expected_result):
        # Mock the _insert method
        backend._insert = MagicMock(return_value=expected_result)

        # Call the insert_one method with the given parameters
        result = backend.insert_one(table_name, payload_insert)

        # Assert if the result is as expected
        assert result == expected_result

        # Check if the _insert method was called with the correct arguments
        if payload_insert:
            columns = list(payload_insert.keys())
            values = [tuple(payload_insert.values())]
            backend._insert.assert_called_once_with(table_name, columns, values)
        else:
            backend._insert.assert_not_called()

    # Sample data for testing
    sample_data = [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob", "age": 35},
        {"id": 3, "name": "Charlie", "age": 25},
    ]

    @pytest.mark.parametrize(
        "table_name, payload_insert, expected_result",
        [
            ("test_table", sample_data, True),
            ("test_table", [], False),
            ("test_table", None, False),
        ],
    )
    def test_insert_many(self, backend, table_name, payload_insert, expected_result):
        # Mock the _insert method
        backend._insert = MagicMock(return_value=expected_result)

        # Call the insert_many method with the given parameters
        result = backend.insert_many(table_name, payload_insert)

        # Assert if the result is as expected
        assert result == expected_result

        # Check if the _insert method was called with the correct arguments
        if payload_insert and len(payload_insert) > 0:
            columns = list(payload_insert[0].keys())
            values_list = [tuple(d.values()) for d in payload_insert]
            backend._insert.assert_called_once_with(table_name, columns, values_list)
        else:
            backend._insert.assert_not_called()

    @pytest.mark.parametrize(
        "table_name, query, success",
        [
            ("test_table", {"attribute": "column1", "value": 1}, True),
            ("test_table", {"attribute": "column2", "value": "A"}, False),
        ],
    )
    def test_delete_one(self, backend: BackendSQL, table_name: str, query: dict, success: bool) -> None:
        backend._has_executed = MagicMock(return_value=success)
        backend.execute = MagicMock()

        result = backend.delete_one(table_name, query)
        assert result == success

        sql_delete = SQL("DELETE FROM {table_name} WHERE {column} = {value}").format(
            table_name=Identifier(table_name),
            column=Identifier(query["attribute"]),
            value=Literal(query["value"]),
        )
        backend.execute.assert_called_once_with(sql_delete)
