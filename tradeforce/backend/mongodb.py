""" backend/mongodb.py

Module: tradeforce.backend.mongodb
----------------------------------

Provides an interface for interacting with MongoDB databases, allowing for operations
such as querying, inserting, updating, and deleting documents. It includes a class,
BackendMongoDB, which extends the Backend class and implements methods specific to MongoDB.

Main methods:

    is_new_entity:  Checks if a collection with the given name exists in the MongoDB
                        database and returns a boolean value indicating the result.

    create_index:   Creates an index on a specified collection with the provided
                        index name and optional uniqueness constraints.

    get_collection: Retrieves a MongoDB Collection object for the given collection name.

    query:          Queries a specified collection in the MongoDB database with provided
                        query parameters, projection, sort order, limit, and skip options,
                        returning a list of query results.

    update_one:     Updates a single document in the specified collection with the provided
                        query parameters, set value, and optional upsert functionality.

    insert_one:     Inserts a single document into the specified collection.

    insert_many:    Inserts multiple documents into the specified collection.

    delete_one:     Deletes a single document from the specified collection using
                        provided query parameters.

A helper function, _construct_mongodb_query, is also provided
to construct MongoDB queries from dictionaries.

"""
from __future__ import annotations
from typing import TYPE_CHECKING
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, CollectionInvalid, OperationFailure, ConfigurationError
from pymongo.collection import Collection
from tradeforce.backend import Backend

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import Tradeforce


def _construct_mongodb_query(query: dict | None) -> dict:
    """Construct a MongoDB query
    -> from the given input dictionary.

    Takes an optional dictionary as a parameter and returns a MongoDB
    query as a dictionary. Supports the "in" operator for checking
    the presence of a value in a list.

    Params:
        query: An optional dictionary containing the query parameters.

    Returns:
        A dictionary representing the MongoDB query.
    """
    if query is None:
        return {}

    has_in_operator = query.get("in", False)

    if has_in_operator:
        mongo_query = {query["attribute"]: {"$in": query["value"]}}
    else:
        mongo_query = {query["attribute"]: query["value"]}

    return mongo_query


class BackendMongoDB(Backend):
    def __init__(self, root: Tradeforce):
        super().__init__(root)
        self.backend_client = self._connect()
        self.is_new_history_entity = self.is_new_entity(self.config.dbms_history_entity_name)

    def _connect(self) -> MongoClient:
        """Connect to the MongoDB instance

        Construct the MongoDB URI, create a MongoClient instance, and establish
        a connection to the MongoDB database. Check the connection using the
        "ping" command.

        Returns:
            A MongoClient instance connected to the MongoDB database.

        Raises:
            ConnectionFailure: If the connection to the MongoDB instance fails.
        """
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

    def is_new_entity(self, collection_name: str) -> bool:
        """Check if the given collection exists.

        Check if a collection with the provided name exists in the MongoDB
        database and return a boolean value indicating the result.

        Note:
            "Entity" is either a SQL table or a MongoDB collection.
            is_new_entity method is also implemented in the BackendSQL interface
            and there refers to a SQL table.

        Params:
            collection_name: A string representing the name of the collection.

        Returns:
            A boolean value indicating whether the collection exists in the MongoDB database.
        """
        collection = self.get_collection(collection_name)

        try:
            self.dbms_db.validate_collection(collection)
        except (CollectionInvalid, OperationFailure):
            return True
        else:
            return False

    # ----------------------------
    # MongoDB specific operations
    # ----------------------------

    def create_index(self, collection_name: str, index_name: str, unique: bool = False) -> None:
        """Create an index on the specified collection.

        Create an index on the given collection with the provided index name.
        The 'unique' parameter indicates whether the index should enforce
        uniqueness constraints.

        Params:
            collection_name: A string representing the name of the collection.
            index_name:      A string representing the name of the index to be created.
            unique:          An optional boolean value indicating whether the index
                                should enforce uniqueness constraints.
        """
        collection = self.get_collection(collection_name)
        collection.create_index(index_name, unique=unique)

    def get_collection(self, collection_name: str) -> Collection:
        """Retrieve a MongoDB collection object.

        Return a Collection object for the given collection name.
        If the collection doesn't exist, it will be created automatically
        by the MongoDB server.

        Params:
            collection_name: A string representing the name of the collection.

        Returns:
            A Collection object representing the specified MongoDB collection.
        """
        if collection_name is None:
            raise ValueError("Provide a collection name. Must not be None!")

        return self.dbms_db[collection_name]

    # ---------------------------------
    # Postgres specific operations
    # ---------------------------------
    # Only here to prevent type errors
    # ---------------------------------

    def db_exists_or_create(self, db_name: str | None = None) -> None:
        """Dummy method to prevent type errors.

        Params:
            db_name: A string representing the name of the database.

        Raises:
            NotImplementedError if called on MongoDB.
        """
        raise NotImplementedError("MongoDB does not support database creation.")

    # ----------------
    # CRUD operations
    # ----------------

    def query(
        self,
        collection_name: str,
        query: dict[str, str | int | list] | None = None,
        projection: dict[str, bool] | None = None,
        sort: list[tuple[str, int]] | None = None,
        limit: int | None = None,
        skip: int | None = None,
    ) -> list:
        """Query the specified collection in the MongoDB database.

        Query the specified collection in the MongoDB database with the provided
        query parameters, projection, sort order, limit, and skip options.

        Params:
            collection_name: A string representing the name of the collection.
            query:           An optional dictionary containing the query parameters.
            projection:      An optional dictionary specifying the fields to return.
            sort:            An optional list of tuples specifying the sort order.
            limit:           An optional integer specifying the maximum number of results to return.
            skip:            An optional integer specifying the number of results to
                                skip before starting to return results.

        Returns:
            A list of query results from the specified MongoDB collection.
        """
        collection = self.get_collection(collection_name)

        mongo_query = _construct_mongodb_query(query)

        if projection is None:
            projection = {"_id": False}
        else:
            projection["_id"] = False

        if skip is None:
            skip = 0

        if limit is None:
            limit = 0

        return list(collection.find(mongo_query, projection=projection, sort=sort, skip=skip, limit=limit))

    def update_one(
        self,
        collection_name: str,
        query: dict[str, str | int | list],
        set_value: str | int | list | dict,
        upsert: bool = False,
    ) -> bool:
        """Update a single document

        Update a single document in the specified collection with the provided
        query parameters, set value, and upsert option.

        Upsert means that if no document matches the query,
        a new document will be created.

        Params:
            collection_name: A string representing the name of the collection.
            query:           A dictionary containing the query parameters.
            set_value:       A value or dictionary to set in the document.
            upsert:          An optional boolean value indicating whether
                                to perform an upsert operation.

        Returns:
            A boolean value indicating the success or failure of the update operation.
        """
        collection = self.get_collection(collection_name)

        mongo_query = _construct_mongodb_query(query)

        try:
            update_result = collection.update_one(mongo_query, {"$set": set_value}, upsert=upsert)
            update_success = update_result.acknowledged
        except (TypeError, ValueError):
            self.log.error("Update into DB failed!")
            update_success = False

        return update_success

    def insert_one(self, collection_name: str, payload_insert: dict) -> bool:
        """Insert a single document.

        Insert a single document into the specified collection.

        Params:
            collection_name: A string representing the name of the collection.
            payload_insert:  A dictionary representing the document to insert.

        Returns:
            A boolean value indicating the success of the insert operation.
        """
        collection = self.get_collection(collection_name)

        try:
            insert_result = collection.insert_one(payload_insert)
            insert_ok = insert_result.acknowledged
        except (TypeError, ConfigurationError):
            insert_ok = False
            self.log.error("Insert into DB failed!")

        return insert_ok

    def insert_many(self, collection_name: str, payload_insert: list[dict]) -> bool:
        """Insert multiple documents.

        Insert multiple documents into the specified collection.

        Params:
            collection_name: A string representing the name of the collection.
            payload_insert:  A list of dictionaries representing the documents to insert.

        Returns:
            A boolean value indicating the success or failure of the insert operation.
        """
        collection = self.get_collection(collection_name)
        try:
            insert_result = collection.insert_many(payload_insert)
            insert_many_ok = insert_result.acknowledged
        except (TypeError, ConfigurationError):
            insert_many_ok = False
            self.log.error("Insert many into DB failed!")
        return insert_many_ok

    def delete_one(self, collection_name: str, query: dict[str, str | int]) -> bool:
        """Delete a single document.

        Delete a single document from the specified collection using the provided
        query parameters.

        Params:
            collection_name: A string representing the name of the collection.
            query:           A dictionary containing the query parameters to
                                identify the document to delete.

        Returns:
            A boolean value indicating the success of the delete operation.
        """

        collection = self.get_collection(collection_name)
        delete_result = collection.delete_one(query)
        delete_ok = delete_result.acknowledged
        return delete_ok

        # TODO WARNING: -> {"asset": order["asset"], "buy_order_id": order["buy_order_id"]}
        # multiple entries not supported yet
