"""_summary_
"""

import pytest

# import numpy as np
# import pandas as pd


# from tradeforce.utils import get_df_datetime_index, drop_dict_na_values
from tradeforce.config import Config
from tradeforce.main import TradingEngine
from tradeforce.backend import Backend

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


@pytest.fixture
def backend():
    return Backend()


def test_init(backend):
    assert backend.is_new_coll_or_table is True
    assert backend.is_filled_na is False


# @pytest.fixture
# def config():
#     return {
#         "dbms": "postgresql",
#         "dbms_user": "username",
#         "dbms_pw": "password",
#         "dbms_host": "localhost",
#         "dbms_port": 5432,
#         "dbms_connect_db": "my_database",
#     }


def test_construct_uri(backend: Backend, config_object: Config):

    # Test with no db_name argument
    expected_uri = "postgresql://username:password@localhost:5432/my_database"
    assert backend.construct_uri(config_object.dbms_db) == expected_uri

    # Test with db_name argument
    expected_uri = "postgresql://username:password@localhost:5432/other_database"
    assert backend.construct_uri("other_database") == expected_uri
