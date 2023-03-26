#################
# Connecting DB #
#################

from __future__ import annotations
import sys
from typing import TYPE_CHECKING
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, CollectionInvalid, OperationFailure, ConfigurationError
from pymongo.collection import Collection
from tradeforce.backend import Backend

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import TradingEngine

# TODO: table_or_coll can be collection or table depending on DBMS. A more general term should be "entity".
# TODO: self.config.use_dbms is not needed. If self.config.dbms is None, then no DBMS is used??


def construct_mongodb_query(query: dict | None) -> dict:
    if query is None:
        return {}
    has_in_operator = query.get("in", False)
    if has_in_operator:
        mongo_query = {query["attribute"]: {"$in": query["value"]}}
    else:
        mongo_query = {query["attribute"]: query["value"]}
    return mongo_query


class BackendMongoDB(Backend):
    def __init__(self, root: TradingEngine):
        super().__init__(root)
        self.backend_client = self.connect()
        self.is_new_coll_or_table = self.check_collection()
        # Only sync backend now if there is no exchange API connection.
        # In case an API connection is used, db_sync_trader_state()
        # will be called once by exchange_ws -> ws_priv_wallet_snapshot()
        if self.config.use_dbms and not self.config.run_live:
            self.db_sync_trader_state()

    def connect(self) -> MongoClient:
        dbms_uri = self.construct_uri()
        dbms_client: MongoClient = MongoClient(dbms_uri, connect=True)
        self.dbms_db = dbms_client[self.config.dbms_db]
        try:
            dbms_client.admin.command("ping")
        except ConnectionFailure as exc:
            self.log.error(
                "Failed connecting to MongoDB (%s). Is the MongoDB instance running and reachable?",
                self.config.dbms_host,
            )
            self.log.exception(exc)
        else:
            self.log.info("Successfully connected to MongoDB backend (%s)", self.config.dbms_host)
        return dbms_client

    def check_collection(self) -> bool:
        collection = self.get_collection(self.config.dbms_table_or_coll_name)
        try:
            self.dbms_db.validate_collection(collection)
        except (CollectionInvalid, OperationFailure):
            is_new_collection = True
        else:
            is_new_collection = False
            if self.config.force_source != "mongodb":
                if self.config.force_source == "api":
                    sys.exit(
                        f"[ERROR] MongoDB history '{self.config.dbms_table_or_coll_name}' already exists. "
                        + "Cannot load history via API. Choose different history DB name or loading method."
                    )
                self.sync_check_needed = True
        return is_new_collection

    ######################
    # MongoDB operations #
    ######################
    def create_index(self, collection_name, index_name, unique=False) -> None:
        collection = self.get_collection(collection_name)
        collection.create_index(index_name, unique=unique)

    def get_collection(self, collection_name) -> Collection:
        if collection_name is None:
            raise ValueError("Provide a collection name. Must not be None!")
        return self.dbms_db[collection_name]

    def query(
        self,
        collection_name: str,
        query: dict[str, str | int | list] | None = None,
        projection: dict[str, bool] | None = None,
        sort: list[tuple[str, int]] | None = None,
        limit: int | None = None,
        skip: int | None = None,
    ) -> list:
        collection = self.get_collection(collection_name)
        mongo_query = construct_mongodb_query(query)
        if projection is None:
            projection = {"_id": False}
        else:
            projection["_id"] = False
        if skip is None:
            skip = 0
        if limit is None:
            limit = 0
        return list(collection.find(mongo_query, projection=projection, sort=sort, skip=skip, limit=limit))
        # return list(collection.find(mongo_query, projection=projection).sort(sort).limit(limit))

    def update_one(
        self, collection_name: str, query: dict[str, str | int | list], set_value: str | int | list | dict, upsert=False
    ) -> bool:

        collection = self.get_collection(collection_name)
        mongo_query = construct_mongodb_query(query)
        try:
            update_result = collection.update_one(mongo_query, {"$set": set_value}, upsert=upsert)
            update_success = update_result.acknowledged
        except (TypeError, ValueError):
            self.log.error("Update into DB failed!")
            update_success = False
        return update_success

    def insert_one(self, collection_name: str, payload_insert: dict) -> bool:
        collection = self.get_collection(collection_name)
        try:
            insert_result = collection.insert_one(payload_insert)
            insert_ok = insert_result.acknowledged
        except (TypeError, ConfigurationError):
            insert_ok = False
            self.log.error("Insert into DB failed!")
        return insert_ok

    def insert_many(self, collection_name: str, payload_insert: list[dict]) -> bool:
        collection = self.get_collection(collection_name)
        try:
            insert_result = collection.insert_many(payload_insert)
            insert_many_ok = insert_result.acknowledged
        except (TypeError, ConfigurationError):
            insert_many_ok = False
            self.log.error("Insert many into DB failed!")
        return insert_many_ok

    def delete_one(self, collection_name, query) -> bool:
        collection = self.get_collection(collection_name)
        delete_result = collection.delete_one(query)
        delete_ok = delete_result.acknowledged
        return delete_ok
        # TODO WARNING: -> {"asset": order["asset"], "buy_order_id": order["buy_order_id"]}
        # multiple entries not supported yet

        # db_acknowledged = (
        #     self.dbms_db[order_type]
        #     .delete_one({"asset": order["asset"], "buy_order_id": order["buy_order_id"]})
        #     .acknowledged
        # )
