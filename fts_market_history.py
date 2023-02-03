"""_summary_
Returns:
    _type_: _description_
"""
import sys
import os
from pathlib import Path
from urllib.parse import quote_plus
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import AutoReconnect, ConnectionFailure, CollectionInvalid, ConfigurationError, OperationFailure

from fts_utils import (
    ms_to_ns,
    get_col_names,
    ns_to_ms,
    get_start_time,
    get_df_datetime_index,
)

from fts_market_metrics import get_init_relevant_assets


def drop_dict_na_values(record):
    return {key: record[key] for key in record if not pd.isna(record[key])}


def get_pct_change(df_history, as_factor=True):
    df_history_pct = df_history.pct_change()
    if as_factor:
        df_history_pct += 1
    df_history_pct.columns = [f"{col}_pct" for col in df_history_pct.columns]
    df_history = pd.concat([df_history, df_history_pct], axis=1, copy=False)
    return df_history_pct


def get_timestamp_intervals(start, end):
    # 50000min == 10000 * 5min -> 10000 max len @ bitfinex api
    timestamp_intervals = [(start, end)]
    if start and end:
        timestamp_list = ns_to_ms(
            pd.date_range(start=ms_to_ns(start), end=ms_to_ns(end), tz="UTC", freq="50000min").asi8
        ).tolist()
        if timestamp_list[-1] != end:
            timestamp_list.append(end)
        timestamp_intervals = list(zip(timestamp_list, timestamp_list[1:]))

    return timestamp_intervals


# TODO: Optimize data types in DBs
# def set_column_type(assets_history_asset, columns_type_int):
#     # Seperate loop because dependend on completion of pct_change
#     for column in assets_history_asset.columns:
#         if column in columns_type_int:
#             assets_history_asset.loc[:, column] = assets_history_asset[column].values.astype(np.int32)
#         else:
#             assets_history_asset.loc[:, column] = assets_history_asset[column].values.astype(np.float32)


def df_fill_na(df_input):
    hist_update_cols = df_input.columns
    volume_cols = hist_update_cols[[("_v" in col) for col in hist_update_cols]]
    ohlc_cols = hist_update_cols[
        [("_o" in col) or ("_h" in col) or ("_l" in col) or ("_c" in col) for col in hist_update_cols]
    ]
    df_input.loc[:, volume_cols] = df_input.loc[:, volume_cols].fillna(0)
    df_input.loc[:, ohlc_cols] = df_input.loc[:, ohlc_cols].fillna(method="ffill")
    df_input.loc[:, ohlc_cols] = df_input.loc[:, ohlc_cols].fillna(method="bfill")


class MarketHistory:
    """Fetch market history from disk and store in DataFrame

    Returns:
        _type_: _description_
    """

    def __init__(self, fts_instance=None, path_current=None, path_history_dumps=None):

        self.fts_instance = fts_instance
        self.config = fts_instance.config
        self.df_market_history = None

        # Set paths
        self.path_current = Path(os.path.dirname(os.path.abspath(__file__))) if path_current is None else path_current
        path_history_dumps = "data/inputs/market_hist_all" if path_history_dumps is None else path_history_dumps
        self.path_feather = (
            self.path_current / f"{path_history_dumps}_{self.config.exchange}_{self.config.base_currency}.arrow"
        )
        self.backend_client = self.connect_backend()
        # State flags
        self.sync_check_needed = False
        self.is_collection_new = True
        self.is_filled_na = False

    def construct_uri(self):
        if self.config.backend_user and self.config.backend_password:
            backend_uri = (
                f"mongodb://{quote_plus(self.config.backend_user)}"
                + f":{quote_plus(self.config.backend_password)}"
                + f"@{self.config.backend_host}"
            )
        else:
            backend_uri = f"mongodb://{self.config.backend_host}"
        return backend_uri

    def db_sync_check(self):
        internal_db_index = np.array(self.get_internal_db_index())

        external_db_index = self.__mongo_exchange_coll.find({}, projection={"_id": False, "t": True}).sort("t", 1)
        external_db_index = np.array([index_dict["t"] for index_dict in list(external_db_index)])

        only_exist_in_external_db = np.setdiff1d(external_db_index, internal_db_index)
        only_exist_in_internal_db = np.setdiff1d(internal_db_index, external_db_index)

        sync_from_external_db_needed = len(only_exist_in_external_db) > 0
        sync_from_internal_db_needed = len(only_exist_in_internal_db) > 0

        if sync_from_external_db_needed:
            external_db_entries_to_sync = self.__mongo_exchange_coll.find(
                {"t": {"$in": only_exist_in_external_db.tolist()}}, projection={"_id": False}
            )
            df_external_db_entries = pd.DataFrame(list(external_db_entries_to_sync)).set_index("t")
            self.df_market_history = pd.concat([self.df_market_history, df_external_db_entries]).sort_values("t")
            print(
                f"[INFO] {len(only_exist_in_external_db)} candles synced from external to internal DB "
                + f"({min(only_exist_in_external_db)} to {max(only_exist_in_external_db)})"
            )

        if sync_from_internal_db_needed:
            internal_db_entries = self.get_market_history(from_list=only_exist_in_internal_db)
            internal_db_entries_to_sync = [
                drop_dict_na_values(record) for record in internal_db_entries.reset_index(drop=False).to_dict("records")
            ]
            self.db_insert(internal_db_entries_to_sync)
            print(
                f"[INFO] {len(only_exist_in_internal_db)} candles synced from internal to external DB "
                + f"({min(only_exist_in_internal_db)} to {max(only_exist_in_internal_db)})"
            )

        print("[INFO] Internal and external DB synced.")
        if self.config.load_history_via == "feather" and sync_from_external_db_needed:
            self.dump_to_feather()

    def connect_backend(self):
        backend_client = None
        if self.config.backend == "mongodb":
            backend_uri = self.construct_uri()
            try:
                backend_client = MongoClient(backend_uri)
                self.backend_db = backend_client[self.config.mongo_exchange_db]
                self.__mongo_exchange_coll = self.backend_db[self.config.mongo_collection]
                backend_client.start_session()
            except (AutoReconnect, ConnectionFailure) as exc:
                print(
                    f"[ERROR] Failed connecting to MongoDB ({self.config.backend_host})."
                    + "Is the MongoDB instance running and reachable?"
                )
                print(exc)
            else:
                print(f"[INFO] Successfully connected to MongoDB backend ({self.config.backend_host})")

            try:
                self.backend_db.validate_collection(self.__mongo_exchange_coll)
            except (CollectionInvalid, OperationFailure):
                self.is_collection_new = True
            else:
                self.is_collection_new = False
                if self.config.load_history_via != "mongodb":
                    if self.config.load_history_via == "api":
                        sys.exit(
                            f"[ERROR] MongoDB history '{self.config.mongo_collection}' already exists. "
                            + "Cannot load history via API. Choose different history DB name or loading method."
                        )
                    self.sync_check_needed = True

        return backend_client

    async def load_history(self, update=False):
        print(f"[INFO] Loading history via {self.config.load_history_via}")
        if self.config.load_history_via == "mongodb":
            cursor = self.__mongo_exchange_coll.find({}, sort=[("t", 1)], projection={"_id": False})
            self.df_market_history = pd.DataFrame(list(cursor))

        elif self.config.load_history_via == "feather":
            self.df_market_history = pd.read_feather(self.path_feather)
            self.df_market_history.set_index("t", inplace=True)
            self.df_market_history.sort_index(inplace=True)

        elif self.config.load_history_via == "api":
            if self.fts_instance.assets_list_symbols is not None:
                end = await self.fts_instance.market_updater_api.get_latest_remote_candle_timestamp()
                start_time = get_start_time(end, delta=self.config.history_timeframe)
                start = start_time["timestamp"]
                await self.update(start=start, end=end)
            else:
                relevant_assets = await get_init_relevant_assets(
                    self.fts_instance, capped=self.config.relevant_assets_cap
                )
                self.fts_instance.assets_list_symbols = relevant_assets["assets"]
                # TODO: Transform into function, already used
                filtered_assets = [
                    f"{asset}_{metric}" for asset in relevant_assets["assets"] for metric in ["o", "h", "l", "c", "v"]
                ]
                await self.update(history_data=relevant_assets["data"][filtered_assets])

                # TODO: Check if "after load" of history is neccessary

                latest_local_candle_timestamp = self.get_local_candle_timestamp(position="latest")
                start_time = get_start_time(latest_local_candle_timestamp, delta=self.config.history_timeframe)
                start = start_time["timestamp"]
                first_local_candle_timestamp = self.get_local_candle_timestamp(position="first")

                print(
                    f"[INFO] Fetching {self.config.history_timeframe} of market history "
                    + f"from {len(self.fts_instance.assets_list_symbols)} assets"
                )
                # TODO: Find better name, not really "start_time" ..
                end_time = get_start_time(first_local_candle_timestamp, delta=self.config.asset_interval)
                end = end_time["timestamp"]
                print(f"{start_time=}")
                print(f"{end_time=}")
                print(f"{start=}")
                print(f"{end=}")
                print(f"{start_time['datetime']=}")
                print(f"{end_time['datetime']=}")

        else:
            sys.exit(
                "[ERROR] load_history_via = '{self.load_history_via}' does not exist. "
                + "Available are 'api', 'mongodb', 'feather' and 'csv'."
            )

        try:
            self.df_market_history.set_index("t", inplace=True)
        except (KeyError, ValueError, TypeError):
            pass

        if self.config.load_history_via == "api" or self.config.load_history_via == "mongodb":
            self.dump_to_feather()

        if self.fts_instance.assets_list_symbols is None:
            self.get_asset_symbols(updated=True)

        amount_assets = len(self.fts_instance.assets_list_symbols)
        print(f"[INFO] {amount_assets} assets from {self.config.exchange} loaded via {self.config.load_history_via}")
        if update:
            await self.update()

        if self.sync_check_needed:
            self.db_sync_check()

        return self.df_market_history

    def dump_to_feather(self):
        self.df_market_history.reset_index(drop=False).to_feather(self.path_feather)
        print(f"[INFO] Assets dumped via feather: {self.path_feather}")

    def get_market_history(
        self,
        assets=None,
        start=None,
        end=None,
        from_list=None,
        latest_candle=False,
        idx_type="loc",
        metrics=None,
        pct_change=False,
        pct_as_factor=True,
        fill_na=False,
        uniform_cols=False,
    ):

        metrics = metrics if metrics else ["o", "h", "l", "c", "v"]
        assets = assets if assets else self.get_asset_symbols(updated=True)

        if latest_candle:
            idx_type = "iloc"
            start = -1

        if from_list is not None:
            idx_type = None

        if start and isinstance(start, str):
            latest_local_candle_timestamp = self.get_local_candle_timestamp(position="latest")
            try:
                start = (pd.to_datetime(latest_local_candle_timestamp, unit="ms", utc=True) - pd.Timedelta(start)).value

            except ValueError as exc:
                raise ValueError(
                    f"ERROR: {start} is not a valid pandas time unit abbreviation!\n"
                    + "For reference check: "
                    + "https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases"
                ) from exc
            else:
                start = ns_to_ms(start)

        # Contruct list of columns to filter
        assets_to_filter = [f"{asset}_{metric}" for asset in assets for metric in metrics]
        self.df_market_history.sort_index(inplace=True)
        if fill_na and not self.is_filled_na:
            df_fill_na(self.df_market_history)
            self.is_filled_na = True

        if idx_type == "loc":
            df_market_history = self.df_market_history.loc[start:end, assets_to_filter]
        if idx_type == "iloc":
            assets_to_filter_idx = [self.df_market_history.columns.tolist().index(asset) for asset in assets_to_filter]
            df_market_history = self.df_market_history.iloc[start:end, assets_to_filter_idx]
        if from_list is not None:
            df_market_history = self.df_market_history.loc[from_list]

        if pct_change:
            df_market_history = get_pct_change(df_market_history, as_factor=pct_as_factor)

        if uniform_cols and len(metrics) == 1:
            df_market_history.columns = get_col_names(df_market_history)
        return df_market_history

    ##################
    # DB interaction #
    ##################

    def db_insert(self, payload_insert, filter_nan=False):
        if filter_nan:
            for entry in payload_insert:
                payload_insert = {items[0]: items[1] for items in entry.items() if pd.notna(items[1])}
        try:
            insert_result = self.__mongo_exchange_coll.insert_many(payload_insert)
            insert_result = insert_result.acknowledged
        except (TypeError, ConfigurationError):
            insert_result = False
            print("[ERROR] Insert into DB failed!")
        return insert_result

    def db_update(self, payload_update, upsert=False, filter_nan=False):
        payload_update_copy = payload_update.copy()
        t_index = payload_update_copy["t"]
        del payload_update_copy["t"]
        if filter_nan:
            payload_update_copy = {items[0]: items[1] for items in payload_update_copy.items() if pd.notna(items[1])}
        try:
            update_result = self.__mongo_exchange_coll.update_one(
                {"t": t_index}, {"$set": payload_update_copy}, upsert=upsert
            )
            update_result = update_result.acknowledged
        except (TypeError, ValueError):
            print("[ERROR] Update into DB failed!")
            update_result = False
        return update_result

    def update_db(self, df_history_update):
        self.df_market_history = pd.concat([self.df_market_history, df_history_update], axis=0).sort_index()
        if self.config.backend == "mongodb":
            db_result = False
            df_history_update.sort_index(inplace=True)
            payload_update = [
                drop_dict_na_values(record) for record in df_history_update.reset_index(drop=False).to_dict("records")
            ]
            if len(payload_update) <= 1:
                db_result = self.db_update(payload_update[0], upsert=True)
            else:
                db_result = self.db_insert(payload_update)
            if db_result and self.is_collection_new:
                self.__mongo_exchange_coll.create_index("t", unique=True)
                self.is_collection_new = False

    async def update(self, history_data=None, start=None, end=None):
        assets_status = None
        if history_data is not None:
            df_history_update = history_data
        else:
            timestamp_intervals = get_timestamp_intervals(start, end)
            df_history_update_list = []
            for interval in timestamp_intervals:
                print(f"[INFO] Fetching history interval: {interval}")
                i_start = interval[0]
                i_end = interval[1]
                df_history_update_interval = await self.fts_instance.market_updater_api.get_market_history(
                    start=i_start, end=i_end
                )
                if df_history_update_interval is None:
                    return assets_status
                df_history_update_list.append(df_history_update_interval)
            df_history_update = pd.concat(df_history_update_list, axis=0)
            df_history_update = df_history_update.iloc[~df_history_update.index.duplicated()]
        self.update_db(df_history_update)
        return assets_status

    def check_db_consistency(self, index=None):
        is_consistent = False
        index = self.get_internal_db_index()
        timeframe = {
            "start_datetime": pd.Timestamp(index[0], unit="ms", tz="UTC"),
            "end_datetime": pd.Timestamp(index[-1], unit="ms", tz="UTC"),
        }
        real_index = get_df_datetime_index(timeframe, freq=self.config.asset_interval)["t"].to_list()
        current_index = index.to_list()
        index_diff = np.setdiff1d(real_index, current_index)
        if len(index_diff) > 0:
            print(f"[WARNING] Inconsistent asset history. Missing candle timestamps: {index_diff}")
        else:
            is_consistent = True
            print("[INFO] Consistency check of DB history successful!")
        return is_consistent

    ###########
    # Getters #
    ###########
    def get_asset_symbols(self, updated=False):
        """Returns a list of all asset symbols

        Returns:
            list: str
        """
        if updated:
            self.fts_instance.assets_list_symbols = get_col_names(self.df_market_history.columns)
        return self.fts_instance.assets_list_symbols

    def get_local_candle_timestamp(self, position="latest", skip=0):
        idx = (-1 - skip) if position == "latest" else (0 + skip)
        latest_candle_timestamp = 0
        if self.df_market_history is not None:
            latest_candle_timestamp = int(self.df_market_history.index[idx])
        elif self.config.backend == "mongodb" and not self.is_collection_new:
            sort_id = -1 if position == "latest" else 1
            cursor = self.__mongo_exchange_coll.find({}, sort=[("t", sort_id)], skip=skip, limit=1)
            latest_candle_timestamp = int(cursor[0]["t"])
        return latest_candle_timestamp

    def get_backend_client(self):
        return self.backend_client

    def get_backend_db(self):
        return self.backend_db

    def get_internal_db_index(self):
        return self.df_market_history.sort_index().index
