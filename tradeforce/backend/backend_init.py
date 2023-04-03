"""
TradeForce Backend Module

This module contains the Backend class for fetching market history from local or remote DBs and
storing it in a DataFrame in-memory for quick internal access. It also provides methods to ensure DB consistency
and synchronization. Class Backend represents the abstract base class for all DB backends
including basic interaction methods like: query, insert, update, delete etc.
The actual DB backends currently supported are MongoDB (class BackendMongoDB) and PostgreSQL (class BackendSQL).
The DBs can be configured in the main config file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from urllib.parse import quote_plus
from tradeforce.market.history import set_internal_db_column_type
from tradeforce.utils import get_reference_index, drop_dict_na_values

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import Tradeforce

############################
# General helper functions #
############################


def get_timeframe_from_index(index: pd.DatetimeIndex) -> dict[str, pd.Timestamp]:
    """Get the start and end datetimes from a given pandas DatetimeIndex.

    Turns a DatetimeIndex into a dictionary containing
    the start and end datetimes.
    """
    return {
        "start_datetime": pd.Timestamp(index[0], unit="ms", tz="UTC"),
        "end_datetime": pd.Timestamp(index[-1], unit="ms", tz="UTC"),
    }


def calculate_index_diff(reference_index: list, current_index: list) -> list:
    """Calculate the difference between the reference index and the current index"""
    return np.setdiff1d(reference_index, current_index).tolist()


def filter_nan_values(payload_update_copy: dict) -> dict:
    """Filter out all NaN values from a given dictionary."""
    return {items[0]: items[1] for items in payload_update_copy.items() if pd.notna(items[1])}


def prepare_payload_update_copy(payload_update: dict, filter_nan: bool) -> tuple[int, dict]:
    """Prepare the payload update copy for the DB update."""
    payload_update_copy = payload_update.copy()
    t_index = int(payload_update_copy["t"])
    del payload_update_copy["t"]
    if filter_nan:  # and self.config.dbms != "postgresql":
        payload_update_copy = filter_nan_values(payload_update_copy)
    return t_index, payload_update_copy


######################
# Backend Base class #
######################


class Backend(ABC):
    """Fetch market history from local or remote database and store in DataFrame"""

    def __init__(self, root: Tradeforce):
        self.root = root
        self.config = root.config
        self.log = root.logging.get_logger(__name__)
        self.is_filled_na = False
        self.reconnect_max_attempts = -1
        self.reconnect_delay_sec = 10

    def construct_uri(self, db_name: str | None = None) -> str:
        """Construct the URI for the DBMS connection."""
        db_name = self.config.dbms_connect_db if db_name is None else db_name
        dbms_uri = (
            f"{self.config.dbms}://"
            + f"{quote_plus(self.config.dbms_user) if self.config.dbms_user else ''}"
            + f"{':' + quote_plus(self.config.dbms_pw) + '@' if self.config.dbms_pw else ''}"
            + f"{self.config.dbms_host}"
            + f":{self.config.dbms_port}"
            + f"/{db_name}"
        )
        return dbms_uri

    ##########################################################################
    # Abstract DB methods:                                                   #
    # Inheriting classes (DB interfaces) must have those methods implemented #
    ##########################################################################

    @abstractmethod
    def query(
        self,
        entity_name: str,
        query: dict[str, str | int | list] | None = None,
        projection: dict[str, bool] | None = None,
        sort: list[tuple[str, int]] | None = None,
        limit: int | None = None,
        skip: int | None = None,
    ) -> list:
        pass

    @abstractmethod
    def update_one(
        self,
        entity_name: str,
        query: dict[str, str | int | list],
        set_value: str | int | list | dict,
        upsert=False,
    ) -> bool:
        pass

    @abstractmethod
    def insert_one(self, entity_name: str, payload_insert: dict) -> bool:
        pass

    @abstractmethod
    def insert_many(self, entity_name: str, payload_insert: list[dict]) -> bool:
        pass

    #########################################
    # Internal in-memory DB related methods #
    #########################################

    def get_internal_db_index(self) -> pd.DatetimeIndex:
        """Get index of internal in-memory DB history in sorted order"""
        return self.root.market_history.internal_history_db.sort_index().index

    ###############################
    # External DB-related methods #
    ###############################

    def get_external_db_index(self) -> np.ndarray:
        query_result = self.query(
            self.config.dbms_history_entity_name,
            projection={"t": True},
            sort=[("t", 1)],
        )
        return np.array([index_dict["t"] for index_dict in query_result])

    ##################################
    # DB consistency and sync checks #
    ##################################

    def check_db_consistency(self) -> bool:
        internal_db_index = self.get_internal_db_index()
        timeframe = get_timeframe_from_index(internal_db_index)
        reference_index = get_reference_index(timeframe, freq=self.config.candle_interval)["t"].to_list()
        index_diff = calculate_index_diff(reference_index, internal_db_index.to_list())

        if len(index_diff) > 0:
            self.log.warning(
                "Inconsistent asset history. Missing %s candle timestamps. Ranging from %s to %s",
                len(index_diff),
                min(index_diff),
                max(index_diff),
            )
            # TODO: fetch missing candle timestamps (index_diff) from remote
            return False

        self.log.info("Consistency check of DB history successful!")
        return True

    def db_sync_check(self) -> None:
        internal_db_index = np.array(self.get_internal_db_index())
        external_db_index = self.get_external_db_index()

        only_exist_in_external_db = np.setdiff1d(external_db_index, internal_db_index)
        only_exist_in_internal_db = np.setdiff1d(internal_db_index, external_db_index)

        if len(only_exist_in_external_db) > 0:
            self.sync_from_external_db(only_exist_in_external_db)

        if len(only_exist_in_internal_db) > 0:
            self.sync_from_internal_db(only_exist_in_internal_db)

        self.log.info("Internal and external DB are synced")
        if self.config.local_cache and len(only_exist_in_external_db) > 0:
            self.root.market_history.save_to_local_cache()

    def sync_from_external_db(self, only_exist_in_external_db: np.ndarray) -> None:
        self.log.info(
            "Syncing %s candles from external to internal DB (%s to %s) ...",
            len(only_exist_in_external_db),
            min(only_exist_in_external_db),
            max(only_exist_in_external_db),
        )
        external_db_entries_to_sync = self.query(
            self.config.dbms_history_entity_name,
            query={
                "attribute": "t",
                "in": True,
                "value": only_exist_in_external_db.tolist(),
            },
        )
        df_external_db_entries = pd.DataFrame(external_db_entries_to_sync).set_index("t")
        df_external_db_entries = set_internal_db_column_type(df_external_db_entries)
        self.root.market_history.internal_history_db = pd.concat(
            [self.root.market_history.internal_history_db, df_external_db_entries]
        ).sort_values("t")

    def sync_from_internal_db(self, only_exist_in_internal_db: np.ndarray) -> None:
        self.log.info(
            "Syncing %s candles from internal to external DB (%s to %s)",
            len(only_exist_in_internal_db),
            min(only_exist_in_internal_db),
            max(only_exist_in_internal_db),
        )
        internal_db_entries = self.root.market_history.get_market_history(from_list=only_exist_in_internal_db)
        internal_db_entries_to_sync: list[dict] = [
            drop_dict_na_values(record, self.config.dbms)
            for record in internal_db_entries.reset_index(drop=False).to_dict("records")
        ]
        self.insert_many(self.config.dbms_history_entity_name, internal_db_entries_to_sync)

    #################################
    # Trader status synchronization #
    #################################

    def db_sync_state_trader(self):
        trader_id = self.config.trader_id
        trader_status = self.get_trader_status(trader_id)

        if trader_status is not None:
            self.sync_existing_trader_status(trader_status)
        else:
            self.create_new_trader_status(trader_id)

    def get_trader_status(self, trader_id: str) -> dict | None:
        db_response = self.query("trader_status", query={"attribute": "trader_id", "value": trader_id})
        return db_response[0] if len(db_response) > 0 and db_response[0]["trader_id"] == trader_id else None

    def sync_existing_trader_status(self, trader_status: dict) -> None:
        self.root.trader.gid = trader_status["gid"]
        if self.config.budget == 0:
            self.config.budget = trader_status["budget"]
        # TODO: Save remaining vals to DB

    def create_new_trader_status(self, trader_id: str) -> None:
        trader_status = {
            "trader_id": trader_id,
            "moving_window_increments": self.config.moving_window_increments,
            "budget": self.config.budget,
            "buy_opportunity_factor": self.config.buy_opportunity_factor,
            "buy_opportunity_boundary": self.config.buy_opportunity_boundary,
            "profit_factor": self.config.profit_factor,
            "amount_invest_fiat": self.config.amount_invest_fiat,
            "maker_fee": self.config.maker_fee,
            "taker_fee": self.config.taker_fee,
            "gid": self.root.trader.gid,
        }
        self.insert_one("trader_status", trader_status)

    def db_sync_state_orders(self):
        trader_id = self.config.trader_id
        self.root.trader.open_orders = self.query("open_orders", query={"attribute": "trader_id", "value": trader_id})
        self.root.trader.closed_orders = self.query(
            "closed_orders", query={"attribute": "trader_id", "value": trader_id}
        )

    ###############################
    # DB update related functions #
    ###############################

    def update_status(self, status_updates):
        for status, value in status_updates.items():
            setattr(self.root.trader, status, value)
        if self.config.use_dbms:
            self.update_one(
                "trader_status",
                query={"attribute": "trader_id", "value": self.config.trader_id},
                set_value=status_updates,
            )

    def db_add_history(self, df_history_update: pd.DataFrame) -> None:
        self.root.market_history.update_internal_db_market_history(df_history_update)
        if self.config.use_dbms:
            payload_update = self.prepare_payload_update(df_history_update.sort_index())
            self.update_or_insert_history(payload_update)

    def prepare_payload_update(self, df_history_update: pd.DataFrame) -> list[dict]:
        return [
            drop_dict_na_values(record, self.config.dbms)
            for record in df_history_update.reset_index(drop=False).to_dict("records")
        ]

    def update_or_insert_history(self, payload_update: list[dict]) -> None:
        if len(payload_update) <= 1:
            self.update_exchange_history(payload_update[0], upsert=True)
        else:
            self.insert_many(self.config.dbms_history_entity_name, payload_update)

    def update_exchange_history(self, payload_update: dict, upsert=False, filter_nan=False) -> bool:
        t_index, payload_update_copy = prepare_payload_update_copy(payload_update, filter_nan)
        update_success = self.update_one(
            self.config.dbms_history_entity_name,
            query={"attribute": "t", "value": t_index},
            set_value=payload_update_copy,
            upsert=upsert,
        )
        return update_success
