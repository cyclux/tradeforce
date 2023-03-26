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
from tradeforce.utils import get_df_datetime_index, drop_dict_na_values

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import TradingEngine
##################
# DB interaction #
##################


class Backend(ABC):
    """Fetch market history from local or remote database and store in DataFrame"""

    def __init__(self, root: TradingEngine):
        self.root = root
        self.config = root.config
        self.log = root.logging.getLogger(__name__)
        # TODO: sync_check_needed always True?
        self.sync_check_needed = True
        self.is_new_coll_or_table = True
        self.is_filled_na = False

    @abstractmethod
    def query(
        self,
        table_or_coll_name: str,
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
        table_or_coll_name: str,
        query: dict[str, str | int | list],
        set_value: str | int | list | dict,
        upsert=False,
    ) -> bool:
        pass

    @abstractmethod
    def insert_one(self, table_or_coll_name: str, payload_insert: dict) -> bool:
        pass

    @abstractmethod
    def insert_many(self, table_or_coll_name: str, payload_insert: list[dict]) -> bool:
        pass

    def construct_uri(self, db_name: str | None = None) -> str:
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

    def get_internal_db_index(self) -> pd.DatetimeIndex:
        return self.root.market_history.df_market_history.sort_index().index

    #############
    # DB checks #
    #############

    def check_db_consistency(self) -> bool:
        is_consistent = False
        index = self.get_internal_db_index()
        timeframe = {
            "start_datetime": pd.Timestamp(index[0], unit="ms", tz="UTC"),
            "end_datetime": pd.Timestamp(index[-1], unit="ms", tz="UTC"),
        }
        real_index = get_df_datetime_index(timeframe, freq=self.config.candle_interval)["t"].to_list()
        current_index = index.to_list()
        index_diff = np.setdiff1d(real_index, current_index)
        if len(index_diff) > 0:
            self.log.warning(
                "Inconsistent asset history. Missing %s candle timestamps: %s", len(index_diff), str(index_diff)
            )
            # TODO: fetch missing candle timestamps (index_diff) from remote
        else:
            is_consistent = True
            self.log.info("Consistency check of DB history successful!")
        return is_consistent

    def db_sync_check(self) -> None:
        internal_db_index = np.array(self.get_internal_db_index())
        # external_db_index =
        # self.exchange_history_table_or_coll.find({}, projection={"_id": False, "t": True}).sort("t", 1)
        # TODO sort by t
        query_result = self.query(self.config.dbms_table_or_coll_name, projection={"t": True}, sort=[("t", 1)])
        external_db_index = np.array([index_dict["t"] for index_dict in query_result])

        only_exist_in_external_db = np.setdiff1d(external_db_index, internal_db_index)
        only_exist_in_internal_db = np.setdiff1d(internal_db_index, external_db_index)

        sync_from_external_db_needed = len(only_exist_in_external_db) > 0
        sync_from_internal_db_needed = len(only_exist_in_internal_db) > 0

        if sync_from_external_db_needed:
            external_db_entries_to_sync = self.query(
                self.config.dbms_table_or_coll_name,
                query={"attribute": "t", "in": True, "value": only_exist_in_external_db.tolist()},
            )
            df_external_db_entries = pd.DataFrame(external_db_entries_to_sync).set_index("t")
            self.root.market_history.df_market_history = pd.concat(
                [self.root.market_history.df_market_history, df_external_db_entries]
            ).sort_values("t")
            self.log.info(
                "%s candles synced from external to internal DB (%s to %s)",
                len(only_exist_in_external_db),
                min(only_exist_in_external_db),
                max(only_exist_in_external_db),
            )

        if sync_from_internal_db_needed:
            internal_db_entries = self.root.market_history.get_market_history(from_list=only_exist_in_internal_db)
            internal_db_entries_to_sync: list[dict] = [
                drop_dict_na_values(record, self.config.dbms)
                for record in internal_db_entries.reset_index(drop=False).to_dict("records")
            ]
            self.insert_many(self.config.dbms_table_or_coll_name, internal_db_entries_to_sync)
            self.log.info(
                "%s candles synced from internal to external DB (%s to %s)",
                len(only_exist_in_internal_db),
                min(only_exist_in_internal_db),
                max(only_exist_in_internal_db),
            )

        self.log.info("Internal and external DB are synced")
        if self.config.local_cache and sync_from_external_db_needed:
            self.root.market_history.save_to_local_cache()

    def db_sync_trader_state(self):
        trader_id = self.config.trader_id
        db_response = self.query("trader_status", query={"attribute": "trader_id", "value": trader_id})
        if len(db_response) > 0 and db_response[0]["trader_id"] == trader_id:
            trader_status = db_response[0]
            self.root.trader.gid = trader_status["gid"]
            if self.config.budget == 0:
                self.config.budget = trader_status["budget"]
            # TODO: Save remaining vals to DB
        else:
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

        # TODO: returns a list of dicts, maybe convert to list, old code did
        self.root.trader.open_orders = self.query("open_orders", query={"attribute": "trader_id", "value": trader_id})

        # TODO: returns a list of dicts, maybe convert to list, old code did
        self.root.trader.closed_orders = self.query(
            "closed_orders", query={"attribute": "trader_id", "value": trader_id}
        )

    ################
    # DB functions #
    ################

    def update_status(self, status_updates):
        for status, value in status_updates.items():
            setattr(self.root.trader, status, value)
        if self.config.use_dbms:
            self.update_one(
                "trader_status",
                query={"attribute": "trader_id", "value": self.config.trader_id},
                set_value=status_updates,
            )

    def db_add_history(self, df_history_update):
        self.root.market_history.df_market_history = pd.concat(
            [self.root.market_history.df_market_history, df_history_update], axis=0
        ).sort_index()
        if self.config.use_dbms:
            db_result_ok = False
            df_history_update.sort_index(inplace=True)
            payload_update = [
                drop_dict_na_values(record, self.config.dbms)
                for record in df_history_update.reset_index(drop=False).to_dict("records")
            ]
            if len(payload_update) <= 1:
                db_result_ok = self.update_exchange_history(payload_update[0], upsert=True)
            else:
                db_result_ok = self.insert_many(self.config.dbms_table_or_coll_name, payload_update)
            # TODO: Check if create_index is needed here
            if db_result_ok and self.is_new_coll_or_table:
                self.create_index(self.config.dbms_table_or_coll_name, "t", unique=True)
                self.is_new_coll_or_table = False

    def update_exchange_history(self, payload_update: dict, upsert=False, filter_nan=False) -> bool:
        payload_update_copy = payload_update.copy()
        t_index = payload_update_copy["t"]
        del payload_update_copy["t"]
        if filter_nan and self.config.dbms != "postgresql":
            payload_update_copy = {items[0]: items[1] for items in payload_update_copy.items() if pd.notna(items[1])}

        update_success = self.update_one(
            self.config.dbms_table_or_coll_name,
            query={"attribute": "t", "value": t_index},
            set_value=payload_update_copy,
            upsert=upsert,
        )
        return update_success
