#################
# Connecting DB #
#################

import sys
import pandas as pd
from urllib.parse import quote_plus
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, AutoReconnect, CollectionInvalid, OperationFailure, ConfigurationError
from tradeforce.backend import Backend

# move to utils
def drop_dict_na_values(record):
    return {key: record[key] for key in record if not pd.isna(record[key])}


class BackendMongoDB(Backend):
    def __init__(self, root):
        super().__init__(root)
        self.backend_client = self.connect_backend_mongo()
        # Only sync backend now if there is no exchange API connection.
        # In case an API connection is used, db_sync_trader_state()
        # will be called once by exchange_ws -> ws_priv_wallet_snapshot()
        if self.config.use_dbms and not self.config.run_live:
            self.db_sync_trader_state()

    def construct_uri_mongo(self):
        if self.config.dbms_user and self.config.dbms_pw:
            dbms_uri = (
                f"mongodb://{quote_plus(self.config.dbms_user)}"
                + f":{quote_plus(self.config.dbms_pw)}"
                + f"@{self.config.dbms_host}"
                + f":{self.config.dbms_port}"
            )
        else:
            dbms_uri = f"mongodb://{self.config.dbms_host}:{self.config.dbms_port}"
        return dbms_uri

    def connect_backend_mongo(self):
        dbms_uri = self.construct_uri_mongo()
        dbms_client = MongoClient(dbms_uri, connect=True)
        self.dbms_db = dbms_client[self.config.dbms_db]
        self.mongo_exchange_coll = self.dbms_db[self.config.dbms_table_or_coll_name]
        try:
            dbms_client.admin.command("ping")
        except (AutoReconnect, ConnectionFailure) as exc:
            self.log.error(
                "Failed connecting to MongoDB (%s). Is the MongoDB instance running and reachable?",
                self.config.dbms_host,
            )
            self.log.exception(exc)
        else:
            self.log.info("Successfully connected to MongoDB backend (%s)", self.config.dbms_host)

        try:
            self.dbms_db.validate_collection(self.mongo_exchange_coll)
        except (CollectionInvalid, OperationFailure):
            self.is_collection_new = True
        else:
            self.is_collection_new = False
            if self.config.force_source != "mongodb":
                if self.config.force_source == "api":
                    sys.exit(
                        f"[ERROR] MongoDB history '{self.config.dbms_table_or_coll_name}' already exists. "
                        + "Cannot load history via API. Choose different history DB name or loading method."
                    )
                self.sync_check_needed = True
        return dbms_client

    def db_sync_trader_state(self):
        trader_id = self.config.trader_id
        db_acknowledged = False
        if self.config.dbms == "mongodb":
            db_response = list(self.dbms_db["trader_status"].find({"trader_id": trader_id}, projection={"_id": False}))
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

                db_acknowledged = self.dbms_db["trader_status"].insert_one(trader_status).acknowledged

            self.root.trader.open_orders = list(
                self.dbms_db["open_orders"].find({"trader_id": trader_id}, projection={"_id": False})
            )
            self.root.trader.closed_orders = list(
                self.dbms_db["closed_orders"].find({"trader_id": trader_id}, projection={"_id": False})
            )
        return db_acknowledged

    ######################
    # MongoDB operations #
    ######################

    def mongodb_insert(self, payload_insert, filter_nan=False):
        if filter_nan:
            for entry in payload_insert:
                payload_insert = {items[0]: items[1] for items in entry.items() if pd.notna(items[1])}
        try:
            insert_result = self.mongo_exchange_coll.insert_many(payload_insert)
            insert_result = insert_result.acknowledged
        except (TypeError, ConfigurationError):
            insert_result = False
            self.log.error("Insert into DB failed!")
        return insert_result

    def mongodb_update(self, payload_update, upsert=False, filter_nan=False):
        payload_update_copy = payload_update.copy()
        t_index = payload_update_copy["t"]
        del payload_update_copy["t"]
        if filter_nan:
            payload_update_copy = {items[0]: items[1] for items in payload_update_copy.items() if pd.notna(items[1])}
        try:
            update_result = self.mongo_exchange_coll.update_one(
                {"t": t_index}, {"$set": payload_update_copy}, upsert=upsert
            )
            update_result = update_result.acknowledged
        except (TypeError, ValueError):
            self.log.error("Update into DB failed!")
            update_result = False
        return update_result

    ####################
    # Order operations #
    ####################

    def order_new(self, order, order_type):
        db_acknowledged = False
        if self.config.dbms == "mongodb":
            db_acknowledged = self.dbms_db[order_type].insert_one(order).acknowledged
        return db_acknowledged

    def order_edit(self, order, order_type):
        db_acknowledged = False
        if self.config.dbms == "mongodb":
            db_acknowledged = (
                self.dbms_db[order_type]
                .update_one({"buy_order_id": order["buy_order_id"]}, {"$set": order})
                .acknowledged
            )
        return db_acknowledged

    def order_del(self, order, order_type):
        db_acknowledged = False
        if self.config.dbms == "mongodb":
            db_acknowledged = (
                self.dbms_db[order_type]
                .delete_one({"asset": order["asset"], "buy_order_id": order["buy_order_id"]})
                .acknowledged
            )
        return db_acknowledged

    ################
    # DB functions #
    ################

    def update_status(self, status_updates):
        db_acknowledged = False
        for status, value in status_updates.items():
            setattr(self.root.trader, status, value)
        if self.config.use_dbms:
            db_acknowledged = (
                self.dbms_db["trader_status"]
                .update_one({"trader_id": self.config.trader_id}, {"$set": status_updates})
                .acknowledged
            )
        return db_acknowledged

    def db_add_history(self, df_history_update):
        self.root.market_history.df_market_history = pd.concat(
            [self.root.market_history.df_market_history, df_history_update], axis=0
        ).sort_index()
        if self.config.dbms == "mongodb":
            db_result = False
            df_history_update.sort_index(inplace=True)
            payload_update = [
                drop_dict_na_values(record) for record in df_history_update.reset_index(drop=False).to_dict("records")
            ]
            if len(payload_update) <= 1:
                db_result = self.mongodb_update(payload_update[0], upsert=True)
            else:
                db_result = self.mongodb_insert(payload_update)
            if db_result and self.is_collection_new:
                self.mongo_exchange_coll.create_index("t", unique=True)
                self.is_collection_new = False
