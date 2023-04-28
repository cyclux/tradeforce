""" backend/backend_core.py

Module: tradeforce.backend.backend_core
---------------------------------------

Contains the Backend class, which serves as a base class for implementing
database backends. Class Backend provides methods for database operations
like querying, updating, inserting, and deleting records, as well as methods
for checking database consistency and synchronizing between internal and
external databases.

Functions:
    get_timeframe_from_index(index: pd.Index) -> dict[str, pd.Timestamp]:
        Get the start and end datetimes from a given pandas DatetimeIndex.

    calculate_index_diff(reference_index: list, current_index: list) -> list:
        Calculate the difference between the reference index and the current index.

    filter_nan_values(payload_update_copy: dict) -> dict:
        Filter out all NaN values from a given dictionary.

    prepare_payload_update_copy(payload_update: dict, filter_nan: bool) -> tuple[int, dict]:
        Prepare the payload update copy for the DB update.

    drop_dict_na_values(record: dict, dbms: str) -> dict:
        Drop all values from dict that are NaN for non-postgresql DBMSs.

Classes:
    Backend(ABC):
        Base class for implementing database backends in the Tradeforce application.
        Provides methods for database operations, checking database consistency, and
        synchronizing data between internal and external databases.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from urllib.parse import quote_plus
from tradeforce.market.history import set_internal_db_column_type
from tradeforce.utils import get_reference_index

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import Tradeforce

# -------------------------
# General helper functions
# -------------------------


def get_timeframe_from_index(index: pd.Index) -> dict[str, pd.Timestamp]:
    """Get the start and end datetimes from a given pandas DatetimeIndex.

    Params:
        index: The DatetimeIndex.

    Returns:
        A dictionary containing the start and end datetimes.
    """
    start = int(index.values[0])
    end = int(index.values[-1])
    return {
        "start_datetime": pd.Timestamp(start, unit="ms", tz="UTC"),
        "end_datetime": pd.Timestamp(end, unit="ms", tz="UTC"),
    }


def calculate_index_diff(reference_index: list, current_index: list) -> list:
    """Calculate the difference between the reference index and the current index.

    Params:
        reference_index: The reference index to compare with.
        current_index: The current index to compare.

    Returns:
        The list of elements that are in the reference_index but not in the current_index.
    """
    return np.setdiff1d(reference_index, current_index).tolist()


def filter_nan_values(payload_update_copy: dict) -> dict:
    """Filter out all NaN values from a given dictionary.

    Params:
        payload_update_copy: The dictionary to filter NaN values from.

    Returns:
        A new dictionary with NaN values removed."""
    return {items[0]: items[1] for items in payload_update_copy.items() if pd.notna(items[1])}


def prepare_payload_update_copy(payload_update: dict, filter_nan: bool) -> tuple[int, dict]:
    """Prepare the payload update copy for the DB update.

    Params:
        payload_update: The payload update to be prepared.
        filter_nan:     A flag indicating if NaN values should be filtered.

    Returns:
        A tuple containing the timestamp index and the prepared payload update copy.
    """
    payload_update_copy = payload_update.copy()
    t_index = int(payload_update_copy["t"])

    del payload_update_copy["t"]

    if filter_nan:  # and self.config.dbms != "postgresql":
        payload_update_copy = filter_nan_values(payload_update_copy)

    return t_index, payload_update_copy


def drop_dict_na_values(record: dict, dbms: str) -> dict:
    """Drop all values from a dict that are NaN.

    Do not drop values for PostgreSQL, because the length of
    all inserts must always be the same.

    Params:
        record: The record to drop NaN values from.
        dbms:   The DBMS in use. For example, 'postgresql' or 'mongodb'.

    Returns:
        The record with all NaN values dropped for MongoDB.
            For PostgreSQL, the original record is returned unchanged.
    """
    if dbms == "postgresql":
        return record
    return {key: record[key] for key in record if not pd.isna(record[key])}


# -------------------
# Backend Base class
# -------------------


class Backend(ABC):
    """Fetch market history from local or remote database and store in DataFrame"""

    def __init__(self, root: Tradeforce):
        """Initialize the Backend class
        -> for fetching market history from local or remote DBs and storing it in a DataFrame.

        Params:
            root: A Tradeforce instance used to access the configuration,
            logging, and other shared resources.

        Attributes:
            root (Tradeforce):            The provided Tradeforce instance.
            config (Config):              The configuration object from the Tradeforce instance.
            log (Logger):                 Logger object for logging messages related to the Backend class.
            is_filled_na (bool):          Flag to indicate if NaN values have been filled.
            reconnect_max_attempts (int): Maximum number of attempts to reconnect to the database.
                                            -1 indicates infinite retries.
            reconnect_delay_sec (int):    Time delay in seconds between reconnection attempts.
        """
        self.root = root
        self.config = root.config
        self.log = root.logging.get_logger(__name__)
        self.is_filled_na = False
        self.reconnect_max_attempts = -1
        self.reconnect_delay_sec = 10

    def construct_uri(self, db_name: str | None = None) -> str:
        """Construct the URI for the DBMS connection.

        Params:
            db_name (optional): The name of the database to connect to.
            If None, uses the default database from the configuration.

        Returns:
            The constructed URI for connecting to the DBMS.
        """
        db_name = self.config.dbms_connect_db if db_name is None else db_name

        return (
            f"{self.config.dbms}://"
            + f"{quote_plus(self.config.dbms_user) if self.config.dbms_user else ''}"
            + f"{':' + quote_plus(self.config.dbms_pw) + '@' if self.config.dbms_pw else ''}"
            + f"{self.config.dbms_host}"
            + f":{str(self.config.dbms_port)}"
            + f"/{db_name}"
        )

    # -------------------------------------
    # Abstract DB methods
    # -------------------------------------
    # Inheriting classes (DB interfaces)
    # must have those methods implemented.
    # -------------------------------------

    @abstractmethod
    def create_index(self, entity_name: str, index_name: str, unique: bool = False) -> None:
        pass

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
        upsert: bool = False,
    ) -> bool:
        pass

    @abstractmethod
    def insert_one(self, entity_name: str, payload_insert: dict) -> bool:
        pass

    @abstractmethod
    def insert_many(self, entity_name: str, payload_insert: list[dict]) -> bool:
        pass

    @abstractmethod
    def delete_one(self, entity_name: str, query: dict[str, str | int]) -> bool:
        pass

    # --------------------------------------
    # Internal in-memory DB related methods
    # --------------------------------------

    def get_internal_db_index(self) -> pd.Index:
        """Get the index of the internal in-memory database history
        -> in sorted order.

        Returns:
            A pandas Index object containing the sorted index of the internal
            in-memory database history.
        """
        return self.root.market_history.internal_history_db.sort_index().index

    # ----------------------------
    # External DB-related methods
    # ----------------------------

    def get_external_db_index(self) -> np.ndarray:
        """Get the index of the external database history.

        Returns:
            An array containing the index of the external database history.
        """
        query_result = self.query(
            self.config.dbms_history_entity_name,
            projection={"t": True},
            sort=[("t", 1)],
        )
        return np.array([index_dict["t"] for index_dict in query_result])

    # -------------------------------
    # DB consistency and sync checks
    # -------------------------------

    def check_db_consistency(self) -> bool:
        """Check the consistency of the internal in-memory database
        -> by comparing it with the reference index.

        Returns:
            True if the internal database history is consistent, otherwise False.

        Raises:
            If False, a warning message will be logged
            with information about the missing candle timestamps.
        """
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
        """Check difference between in-memory and external database

        and synchronize them if needed:

            If there are records that exist only in the external database, this
            method will call the 'sync_from_external_db'
            method to synchronize the internal database with the external one.

            If there are records that exist only in the internal database, this
            method will call the sync_from_internal_db method to synchronize the
            external database with the internal one.

        If the 'local_cache' configuration is set to True and there are records
        that have been synced from the external database, the market history
        will be saved to the local cache.
        """
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
        """Synchronize the internal in-memory database with the external database

        by importing the records that only exist in the external database:
        Receives an array of timestamps that only exist in the external database.
        Query the external database to fetch the corresponding records, and then
        convert the result into a DataFrame, setting the "t" column as the index
        and apply the appropriate column data types.

        Finally, concatenate the retrieved DataFrame with the 'internal_history_db'
        DataFrame and sort the resulting DataFrame by the "t" column. An info log
        message is generated, indicating the number of candles synced and their
        timestamp range.
        """
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
        """Synchronize the external database with the internal in-memory database

        by exporting the records that only exist in the internal database. Receives
        an array of timestamps that only exist in the internal database: Fetch the
        corresponding records from the in-memory database and convert the result
        into a list of dictionaries, dropping any NaN values if the DBMS in use is
        not PostgreSQL.

        Finally, insert the list of dictionaries into the external database using
        the insert_many() method.
        """
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

    # ------------------------------
    # Trader status synchronization
    # ------------------------------

    def db_sync_state_trader(self) -> None:
        """Synchronize the trader state with the external database.

        Retrieves the trader status using the get_trader_status() method.

        If the trader status exists in the external database, call the
        sync_existing_trader_status() method to synchronize the existing
        trader status.

        If not, create a new trader status in the external
        database using the create_new_trader_status() method.
        """
        trader_id = self.config.trader_id
        trader_status = self.get_trader_status(trader_id)

        if trader_status is not None:
            self.sync_existing_trader_status(trader_status)
        else:
            self.create_new_trader_status(trader_id)

    def get_trader_status(self, trader_id: int) -> dict | None:
        """Retrieve the trader status from the external database
        -> based on the trader_id.

        Params:
            trader_id: The unique identifier of the trader.

        Returns:
            A dictionary containing the trader status if found in
            the external database, otherwise None.
        """
        db_response = self.query("trader_status", query={"attribute": "trader_id", "value": trader_id})
        return db_response[0] if len(db_response) > 0 and db_response[0]["trader_id"] == trader_id else None

    def sync_existing_trader_status(self, trader_status: dict) -> None:
        """Synchronize the existing trader status with the external database.

        Update the gid and budget attributes of the trader and the config objects
        with the values retrieved from the 'trader_status' dictionary. If the
        'config.budget' is set to 0, update with the value from the 'trader_status'.

        Params:
            trader_status: A dictionary containing the trader status information.
        """
        self.root.trader.gid = trader_status["gid"]

        if self.config.budget == 0:
            self.config.budget = trader_status["budget"]
        # TODO: Save remaining vals to DB

    def create_new_trader_status(self, trader_id: int) -> None:
        """Create a new trader status in the external database.

        Create a new trader status entry with the provided 'trader_id' and
        other configuration values, then insert the 'trader_status' dictionary
        into the external database.

        Params:
            trader_id: The unique identifier of the trader.
        """
        trader_status = {
            "trader_id": trader_id,
            "moving_window_increments": self.config.moving_window_increments,
            "budget": self.config.budget,
            "buy_signal_score": self.config.buy_signal_score,
            "buy_signal_boundary": self.config.buy_signal_boundary,
            "profit_factor_target": self.config.profit_factor_target,
            "amount_invest_per_asset": self.config.amount_invest_per_asset,
            "maker_fee": self.config.maker_fee,
            "taker_fee": self.config.taker_fee,
            "gid": self.root.trader.gid,
        }
        self.insert_one("trader_status", trader_status)

    def db_sync_state_orders(self) -> None:
        """Sync the state of open and closed orders with the external database.

        Retrieves open and closed orders associated with the 'trader_id' from the
        external database: Assign them to the trader's 'open_orders' and
        'closed_orders' attributes, respectively.
        """
        trader_id = self.config.trader_id
        self.root.trader.open_orders = self.query("open_orders", query={"attribute": "trader_id", "value": trader_id})
        self.root.trader.closed_orders = self.query(
            "closed_orders", query={"attribute": "trader_id", "value": trader_id}
        )

    # ----------------------------
    # DB update related functions
    # ----------------------------

    def update_status(self, status_updates: dict) -> None:
        """Create a new trader status in the external database.

        Create a new trader status entry with the provided trader_id and other config
        values, then insert the trader_status dictionary into the external database.

        Params:
            trader_id: The unique identifier of the trader.
        """
        for status, value in status_updates.items():
            setattr(self.root.trader, status, value)

        if self.config.use_dbms:
            self.update_one(
                "trader_status",
                query={"attribute": "trader_id", "value": self.config.trader_id},
                set_value=status_updates,
            )

    def db_add_history(self, df_history_update: pd.DataFrame) -> None:
        """Add new history data to the internal market history DataFrame
        -> and synchronize with the external database.

        Update the internal market history with the given DataFrame and,
        if the 'use_dbms' flag is enabled, prepare and insert or update
        the external database accordingly.

        Params:
            df_history_update: A DataFrame containing the new history data.
        """
        self.root.market_history.update_internal_db_market_history(df_history_update)

        if self.config.use_dbms:
            payload_update = self.prepare_payload_update(df_history_update.sort_index())
            self.update_or_insert_history(payload_update)

    def prepare_payload_update(self, df_history_update: pd.DataFrame) -> list[dict]:
        """Prepare the payload update
        -> for inserting or updating the external database with new history data.

        Converts the given DataFrame into a list of dictionaries and drop any NaN
        values from the records based on the DBMS configuration.

        Params:
            df_history_update: A DataFrame containing the new history data.

        Returns:c
            A list of dictionaries representing the payload update, with NaN values
            dropped based on the DBMS configuration.
        """
        return [
            drop_dict_na_values(record, self.config.dbms)
            for record in df_history_update.reset_index(drop=False).to_dict("records")
        ]

    def update_or_insert_history(self, payload_update: list[dict]) -> None:
        """Update or insert new history data in the external database
        -> based on the given payload update.

        Updates the external database with the provided payload update:

            - If there is only one record in the payload update, perform an
                upsert operation (update or insert).

            - Otherwise, insert multiple records at once.

        Params:
            payload_update: A list of dictionaries containing the new history data.
        """
        if len(payload_update) <= 1:
            self.update_exchange_history(payload_update[0], upsert=True)
        else:
            self.insert_many(self.config.dbms_history_entity_name, payload_update)

    def update_exchange_history(self, payload_update: dict, upsert: bool = False, filter_nan: bool = False) -> bool:
        """Update the exchange history in the external database
        -> with the given payload update.

        Update the external database with the provided payload for a specific timestamp index.
        If the upsert flag is enabled, insert the record if it doesn't already exist.

        The 'filter_nan' flag determines whether to filter out NaN values from the payload.

        Params:
            payload_update: A dictionary containing the new history data.
            upsert:         A boolean flag indicating whether to perform an upsert operation.
            filter_nan:     A boolean flag indicating whether to filter out NaNs from the payload.

        Returns:
            A boolean value indicating whether the update operation was successful.
        """
        t_index, payload_update_copy = prepare_payload_update_copy(payload_update, filter_nan)

        update_success = self.update_one(
            self.config.dbms_history_entity_name,
            query={"attribute": "t", "value": t_index},
            set_value=payload_update_copy,
            upsert=upsert,
        )

        return update_success
