# mypy: disable-error-code="assignment, attr-defined, misc"

import pytest
from psycopg2 import OperationalError
from psycopg2.sql import SQL, Identifier, Literal, Composed
from unittest.mock import MagicMock
from tradeforce.config import Config
from tradeforce import TradingEngine
from tradeforce.backend import BackendSQL
from tradeforce.backend.sql_tables import CreateTables

config = {
    "trader": {
        "id": 3,
        "run_live": False,
        "creds_path": "exchange_creds.cfg",
        "budget": 0,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
            "amount_invest_fiat": 1000,
            "investment_cap": 0,
            "buy_opportunity_factor": 0.10,
            "buy_opportunity_boundary": 0.05,
            "prefer_performance": 1,
            "hold_time_limit": 1000,
            "profit_factor": 1.10,
            "profit_ratio_limit": 1.01,
            "moving_window_hours": 160,
        },
    },
    "backend": {
        "dbms": "postgresql",
        "dbms_host": "localhost",
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
        "history_timeframe": "60days",
        "update_mode": "live",  # none, once or live
        # "force_source": "postgresql",
    },
}


@pytest.fixture
def mock_config() -> dict:
    return config


@pytest.fixture
def trading_engine(mock_config: dict) -> TradingEngine:
    return TradingEngine(config=mock_config)


@pytest.fixture
def config_object(trading_engine: TradingEngine, mock_config: dict) -> Config:
    return Config(root=trading_engine, config_input=mock_config)


class TestBackendSQL:
    @pytest.fixture()
    def backend(self, trading_engine: TradingEngine) -> BackendSQL:
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

    def test_init(self, backend: BackendSQL, config_object: Config) -> None:
        assert backend.connected is True
        assert isinstance(backend.create_table, CreateTables)
        assert backend.config.dbms_db == config_object.dbms_db

    @pytest.mark.parametrize("run_live, call_db_sync_state_trader", [(True, True), (False, False)])
    def test_init_sync(
        self, backend: BackendSQL, trading_engine: TradingEngine, run_live: bool, call_db_sync_state_trader: bool
    ) -> None:
        backend.config.run_live = run_live
        backend._establish_connection = MagicMock(name="_establish_connection", return_value=None)
        backend.__init__(trading_engine)
        if call_db_sync_state_trader:
            assert backend.config.run_live is True
            backend.db_sync_state_trader.assert_not_called()
            backend.db_sync_state_orders.assert_not_called()

        else:
            assert backend.config.run_live is False
            backend.db_sync_state_trader.assert_called_once()
            backend.db_sync_state_orders.assert_called_once()

    def test_connect_success(self, backend: BackendSQL):
        # Test a successful connection to the database
        backend._connect("postgres")
        assert backend.connected is True

    def test_connect_failure(self, backend: BackendSQL):
        # Test a failed connection to the database
        backend._connect("non_existent_db")
        assert backend.connected is False

    def test_establish_connection_success(self, backend: BackendSQL):
        # Test a successful initialization of the connection to the database if the database already exists
        backend._establish_connection(backend.config.dbms_db)
        assert backend.connected is True

    def test_establish_connection_failure(self, backend: BackendSQL, config_object: Config):
        # Test a successful initialization of the connection to the database if the database does not exist
        # A connection to the postgres database will be established and the database will be created
        db_name = "non_existent_db"
        backend._establish_connection(db_name)
        assert backend.connected is True
        assert backend.config.dbms_db == config_object.dbms_db

    def test_establish_connection_db_name_dbms_db_is_new_coll_or_table_true(
        self, backend: BackendSQL, config_object: Config
    ):
        db_name = config_object.dbms_db
        backend._check_table = MagicMock(name="check_table", return_value=True)
        backend._establish_connection(db_name)
        backend.create_table.trader_status.assert_called_once()
        backend.create_table.open_orders.assert_called_once()
        backend.create_table.closed_orders.assert_called_once()

    def test_establish_connection_db_name_dbms_db_is_new_coll_or_table_false(
        self, backend: BackendSQL, config_object: Config
    ):
        db_name = config_object.dbms_db
        backend._check_table = MagicMock(name="check_table", return_value=False)
        backend._establish_connection(db_name)
        backend.create_table.trader_status.assert_not_called()
        backend.create_table.open_orders.assert_not_called()
        backend.create_table.closed_orders.assert_not_called()

    def test_execute_successful(self, backend: BackendSQL):
        query = SQL("SELECT * FROM {}").format(Identifier("postgres"))
        assert backend.execute(query) is True
        backend.dbms_db.execute.assert_called_once_with(query)

    def test_execute_failed(self, backend: BackendSQL):
        query = SQL("SELECT * FROM {}").format(Identifier("postgres"))
        backend.dbms_db.execute.side_effect = [TypeError("execute failed"), OperationalError("execute failed")]
        assert backend.execute(query) is False
        backend.dbms_db.execute.assert_called_once_with(query)
        backend.log.error.assert_called_once_with("SQL execute failed!")

    def test_db_exists_or_create(self, backend: BackendSQL):
        # Test the db_exists_or_create function
        backend.dbms_db.fetchone.side_effect = [None, (1,)]

        # Case 1: Database does not exist
        backend._db_exists_or_create()
        query1 = SQL("SELECT 1 FROM pg_catalog.pg_database WHERE datname = {db_name};").format(
            db_name=Literal(backend.config.dbms_db)
        )
        query2 = SQL("CREATE DATABASE {db_name};").format(db_name=Identifier(backend.config.dbms_db))

        # Assertions
        backend.dbms_db.execute.assert_any_call(query1)
        backend.dbms_db.execute.assert_any_call(query2)
        backend.log.info.assert_called_with("Created database %s", backend.config.dbms_db)

        # Reset log and execute call count
        backend.log.reset_mock()
        backend.dbms_db.execute.reset_mock()

        # Case 2: Database exists
        backend._db_exists_or_create()

        # Assertions
        backend.dbms_db.execute.assert_called_with(query1)
        backend.log.info.assert_called_with("Database %s already exists", backend.config.dbms_db)

    def test_check_table(self, backend: BackendSQL):
        # Test the check_table function
        backend.dbms_db.fetchone.side_effect = iter([None, (1,)] * 4)

        # Case 1: Table does not exist
        is_new_table = backend._check_table()
        assert is_new_table is True
        query1 = SQL("SELECT 1 FROM pg_catalog.pg_tables WHERE tablename = {table_name};").format(
            table_name=Literal(backend.config.dbms_entity_name)
        )
        # Assertion
        backend.dbms_db.execute.assert_called_with(query1)

        # Reset execute call count
        backend.dbms_db.execute.reset_mock()

        # Case 2: Table exists
        is_new_table = backend._check_table()

        # Assertions
        assert is_new_table is False
        backend.dbms_db.execute.assert_called_with(query1)

        # Reset execute call count
        backend.dbms_db.execute.reset_mock()

        # Case 3: Table exists, force_source == "api"
        backend.config.force_source = "api"
        backend.dbms_db.fetchone.side_effect = iter([(1,)])
        with pytest.raises(SystemExit) as sys_exit:
            backend._check_table()

        # Assertions
        assert sys_exit.type == SystemExit
        backend.dbms_db.execute.assert_called_with(query1)

        # Reset execute call count
        backend.dbms_db.execute.reset_mock()

        # Case 4: Table exists, force_source != "postgresql" and force_source != "api"
        backend.config.force_source = "other"
        backend.dbms_db.fetchone.side_effect = iter([(1,)])
        is_new_table = backend._check_table()

        # Assertions
        assert is_new_table is False
        assert backend.sync_check_needed is True
        backend.dbms_db.execute.assert_called_with(query1)

    @pytest.mark.parametrize("unique", [True, False])
    def test_create_index(self, backend: BackendSQL, unique: bool):
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
    def test__insert(
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

    # Tests for insert_one
    @pytest.mark.parametrize(
        "table_name, payload_insert, _insert_result, expected_result",
        [
            ("test_table", {"column1": 1, "column2": "A"}, True, True),
            ("test_table", {}, False, False),
        ],
    )
    def test_insert_one(
        self,
        backend: BackendSQL,
        table_name: str,
        payload_insert: dict,
        _insert_result: bool,
        expected_result: bool,
    ):
        backend._insert = MagicMock(return_value=_insert_result)
        backend.log.warning = MagicMock()

        result = backend.insert_one(table_name, payload_insert)
        assert result == expected_result

        if payload_insert:
            columns = list(payload_insert.keys())
            values = [tuple(payload_insert.values())]
            backend._insert.assert_called_once_with(table_name, columns, values)
            backend.log.warning.assert_not_called()
        else:
            backend._insert.assert_not_called()
            backend.log.warning.assert_called_once_with(
                "No data to insert into DB! [table: {table_name}]", table_name=table_name
            )

    # Tests for insert_many
    @pytest.mark.parametrize(
        "table_name, payload_insert, _insert_result, expected_result",
        [
            ("test_table", [{"column1": 1, "column2": "A"}, {"column1": 2, "column2": "B"}], True, True),
            ("test_table", [], False, False),
        ],
    )
    def test_insert_many(
        self,
        backend: BackendSQL,
        table_name: str,
        payload_insert: list[dict],
        _insert_result: bool,
        expected_result: bool,
    ):
        backend._insert = MagicMock(return_value=_insert_result)
        backend.log.warning = MagicMock()

        result = backend.insert_many(table_name, payload_insert)
        assert result == expected_result

        if payload_insert:
            columns = list(payload_insert[0].keys())
            values_list = [tuple(d.values()) for d in payload_insert]
            backend._insert.assert_called_once_with(table_name, columns, values_list)
            backend.log.warning.assert_not_called()
        else:
            backend._insert.assert_not_called()
            backend.log.warning.assert_called_once_with(
                "No data to insert into DB! [table: {table_name}]", table_name=table_name
            )

    @pytest.mark.parametrize(
        "table_name, query, success",
        [
            ("test_table", {"attribute": "column1", "value": 1}, True),
            ("test_table", {"attribute": "column2", "value": "A"}, False),
        ],
    )
    def test_delete_one(self, backend: BackendSQL, table_name: str, query: dict, success: bool):
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
