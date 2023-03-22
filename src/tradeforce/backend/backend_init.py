"""_summary_

Returns:
    _type_: _description_
"""

from __future__ import annotations
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


class Backend:
    """Fetch market history from local or remote database and store in DataFrame"""

    def __init__(self, root: TradingEngine):
        self.root = root
        self.config = root.config
        self.log = root.logging.getLogger(__name__)
        # TODO: sync_check_needed always True?
        self.sync_check_needed = True
        self.is_new_coll_or_table = True
        self.is_filled_na = False

    def construct_uri(self, db_name=None):
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

    def get_internal_db_index(self):
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
            self.log.warning("Inconsistent asset history. Missing candle timestamps: %s", str(index_diff))
            # TODO: fetch missing candle timestamps (index_diff) from remote
        else:
            is_consistent = True
            self.log.info("Consistency check of DB history successful!")
        return is_consistent

    def db_sync_check(self) -> None:
        internal_db_index = np.array(self.get_internal_db_index())
        # external_db_index =
        # self.exchange_history_table_or_coll.find({}, projection={"_id": False, "t": True}).sort("t", 1)
        external_db_index = self.query(self.config.dbms_table_or_coll_name, projection={"t": True})  # type: ignore
        external_db_index = np.array([index_dict["t"] for index_dict in external_db_index])

        only_exist_in_external_db = np.setdiff1d(external_db_index, internal_db_index)
        only_exist_in_internal_db = np.setdiff1d(internal_db_index, external_db_index)

        sync_from_external_db_needed = len(only_exist_in_external_db) > 0
        sync_from_internal_db_needed = len(only_exist_in_internal_db) > 0

        if sync_from_external_db_needed:
            # external_db_entries_to_sync = self.exchange_history_table_or_coll.find(
            #     {"t": {"$in": only_exist_in_external_db.tolist()}}, projection={"_id": False}
            # )
            # external_db_entries_to_sync = self.query(
            #     self.config.dbms_table_or_coll_name, query={"t": {"$in": only_exist_in_external_db.tolist()}}
            # )
            external_db_entries_to_sync = self.query(  # type: ignore
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
            self.insert_many(self.config.dbms_table_or_coll_name, internal_db_entries_to_sync)  # type: ignore
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
        # db_response = list(self.dbms_db["trader_status"].find({"trader_id": trader_id}, projection={"_id": False}))
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
            # db_acknowledged = self.dbms_db["trader_status"].insert_one(trader_status).acknowledged
            self.insert_one("trader_status", trader_status)

        # self.root.trader.open_orders = list(
        #     self.dbms_db["open_orders"].find({"trader_id": trader_id}, projection={"_id": False})
        # )
        self.root.trader.open_orders = self.query("open_orders", query={"attribute": "trader_id", "value": trader_id})

        # self.root.trader.closed_orders = list(
        #     self.dbms_db["closed_orders"].find({"trader_id": trader_id}, projection={"_id": False})
        # )
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
            # db_acknowledged = (
            #     self.dbms_db["trader_status"]
            #     .update_one({"trader_id": self.config.trader_id}, {"$set": status_updates})
            #     .acknowledged
            # )

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

    def update_exchange_history(self, payload_update, upsert=False, filter_nan=False):
        payload_update_copy = payload_update.copy()
        t_index = payload_update_copy["t"]
        del payload_update_copy["t"]
        if filter_nan:
            payload_update_copy = {items[0]: items[1] for items in payload_update_copy.items() if pd.notna(items[1])}
        # try:
        #     update_result = self.dbms_db[self.config.dbms_table_or_coll_name].update_one(
        #         {"t": t_index}, {"$set": payload_update_copy}, upsert=upsert
        #     )
        #     update_result = update_result.acknowledged
        # except (TypeError, ValueError):
        #     self.log.error("Update into DB failed!")
        #     update_result = False
        update_success = self.update_one(
            self.config.dbms_table_or_coll_name,
            query={"attribute": "t", "value": t_index},
            set_value=payload_update_copy,
            upsert=upsert,
        )
        return update_success
