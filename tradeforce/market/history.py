""" market/history.py

Module: tradeforce.market.history
---------------------------------

Provides the MarketHistory class, which is responsible for loading,
fetching, updating, and retrieving market history data for a list of specified
assets from a configured exchange or database backend.

It supports loading market history data from local cache, backend database
(MongoDB or PostgreSQL), and the exchange API. The loading process is performed
in a sequence, where if data is not found or not loadable from one source, it
will attempt to load from the next source in the sequence.

The MarketHistory class also provides methods to perform post-processing steps,
saving data to local cache, updating the market history with new data, and
retrieving (filtered subsets of) the market history data as a DataFrame.

Main MarketHistory class methods:
    update_internal_db_market_history: Updates the internal history database
                                        with new market history data provided as a DataFrame.

    load_history:                      Loads market history data from available sources
                                        (local cache, backend database, or API) based on the configuration settings
                                        and updates the internal history database accordingly.

    save_to_local_cache:               Saves the internal history database as a
                                        Feather file in the local cache directory.

    get_market_history:                Retrieves a filtered subset of the market history data as a DataFrame,
                                        based on the provided parameters, such as assets, time range, and metrics.

    update:                            Updates the market history with new data, either provided
                                        as a DataFrame or fetched from the API within a specified time range,
                                        and updates the backend database accordingly.

    get_asset_symbols:                 Retrieves a list of all asset symbols in the internal history database.
                                        If the updated parameter is set to True, the asset symbols list is updated
                                        by extracting them from the internal history database.

    get_init_relevant_assets:          Fetches the initial relevant assets based on the root configuration and the
                                        relevant assets cap, then updates the global asset symbols.

Functions:
    _get_timestamp_intervals(start: int, end: int, interval: str) -> list[tuple[int, int]]:
    Calculates the timestamp intervals based on the provided start, end, and interval.

    _filter_assets(df: pd.DataFrame, assets: list[str], metrics: list[str]) -> pd.DataFrame:
        Filters a DataFrame to include only the specified assets and metrics.

    _iloc_filter_assets(df: pd.DataFrame, assets_to_filter: list[str]) -> pd.DataFrame:
        Filters a DataFrame to include only the specified assets using iloc.

    _get_pct_change(df: pd.DataFrame, first_row: int | float | None, as_factor: bool) -> pd.DataFrame:
        Calculates the percentage change in the values of a DataFrame.

    _uniform_columns(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        Returns a DataFrame with uniform column naming.

    get_col_names(columns: pd.Index) -> list[str]:
        Extracts and returns the asset symbols from column names in a DataFrame.

    set_internal_db_column_type(df: pd.DataFrame) -> pd.DataFrame:
        Sets the column types of a DataFrame to float.

    df_fill_na(df: pd.DataFrame) -> None:
        Fills NaN values in a DataFrame.

    get_init_relevant_assets(root: Root, capped: int | None = None) -> DictRelevantAssets:
        Fetches the initial relevant assets based on the root configuration and the relevant assets cap.

"""


from __future__ import annotations
from pathlib import Path
import asyncio

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING
from tradeforce.custom_types import DictRelevantAssets
from tradeforce.utils import ms_to_ns, get_col_names, ns_to_ms_array, get_time_minus_delta, candle_interval_to_ms
from tradeforce.market.metrics import get_init_relevant_assets

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce import Tradeforce


def _get_pct_change(
    df_history: pd.DataFrame, pct_first_row: int | float | None, as_factor: bool = True
) -> pd.DataFrame:
    """Calculate the percentage change between each row and the previous row of a DataFrame.

    Optionally, convert the changes to factors by adding 1 to each value (e.g. -0.1 -> 0.9).
    Allows setting a specific value for the first row of the output DataFrame because
    pct_change() returns NaN for the first row, which is not always desired.
    Also renames the columns of the output DataFrame to include "_pct" (e.g. "BTC_o" -> "BTC_o_pct").

    Params:
        df_history: The input DataFrame containing market history data.
        pct_first_row: The value to be set for the first row of the output DataFrame.
        as_factor: If True, converts percentage changes to factors. Default is True.

    Returns:
        A DataFrame containing the percentage changes (or factors) for each row compared
            to the previous row, with updated column names.
    """
    df_history_pct = df_history.pct_change()

    if as_factor:
        df_history_pct += 1

    if pct_first_row is not None:
        df_history_pct.iloc[0] = pct_first_row

    df_history_pct.columns = pd.Index([f"{col}_pct" for col in df_history_pct.columns])
    return df_history_pct


def _get_timestamp_intervals(start: int, end: int, candle_interval: str) -> list[tuple[int, int]]:
    """Generate a list of timestamp intervals given a start and end timestamp and a candle interval.

    The intervals are created to have a maximum length of 10,000, which is the maximum
    number of data points allowed in a single request by the Bitfinex API.

    Params:
        start:           The start timestamp of the range.
        end:             The end timestamp of the range.
        candle_interval: The interval between each candle in the format accepted by pandas (e.g., '5min').

    Returns:
        A list of tuples, where each tuple represents an interval (start, end)
        between the given start and end timestamps.
    """
    timestamp_intervals = [(-1, -1)]

    if start != -1 and end != -1:
        freq_for_max_request = pd.Timedelta(candle_interval) * 10_000

        start_dt = pd.to_datetime(ms_to_ns(start))
        end_dt = pd.to_datetime(ms_to_ns(end))

        timestamp_array = ns_to_ms_array(pd.date_range(start_dt, end_dt, tz="UTC", freq=freq_for_max_request).asi8)
        timestamp_list: list = np.array(timestamp_array).tolist()
        if timestamp_list[-1] != end:
            timestamp_list.append(end)

        timestamp_intervals = list(zip(timestamp_list, timestamp_list[1:]))

    return timestamp_intervals


def set_internal_db_column_type(inmemory_db: pd.DataFrame) -> pd.DataFrame:
    """Set the internal database column types to float32 for optimized memory usage.

    Params:
        inmemory_db: The input DataFrame representing the in-memory database.

    Returns:
        A DataFrame with the same data as the input but with all column types set to float32.
    """
    return inmemory_db.astype(np.float32)


def _df_fill_na(df_input: pd.DataFrame) -> None:
    """Fill NaN values in a DataFrame containing market history data.

    For volume columns (those with "_v" in the column name), NaN values are replaced with 0.
    For OHLC columns (those with "_o", "_h", "_l", or "_c" in the column name),
    NaN values are filled using forward-fill and then back-fill methods.

    Note that this function modifies the input DataFrame in-place.

    Params:
        df_input: The input DataFrame containing market history data with NaN values.
    """
    hist_update_cols = df_input.columns

    volume_cols = pd.Index([col for col in hist_update_cols if "_v" in col])
    ohlc_cols = pd.Index([col for col in hist_update_cols if any(x in col for x in ["_o", "_h", "_l", "_c"])])

    # Mypy: .loc indexing not recognized correctly
    df_input.loc[:, volume_cols] = df_input.loc[:, volume_cols].fillna(0)  # type: ignore[index]
    df_input.loc[:, ohlc_cols] = df_input.loc[:, ohlc_cols].fillna(method="ffill")  # type: ignore[index]
    df_input.loc[:, ohlc_cols] = df_input.loc[:, ohlc_cols].fillna(method="bfill")  # type: ignore[index]


def _filter_assets(internal_history_db: pd.DataFrame, assets: list[str], metrics: list) -> pd.DataFrame:
    """Filter the in-memory DB by the given assets and metrics.

    Params:
        internal_history_db: The internal history database DataFrame.
        assets:              A list of asset symbols to filter by.
        metrics:             A list of metrics to filter by.

    Returns:
        A DataFrame with columns filtered by the given assets and metrics.
    """
    assets_to_filter = [f"{asset}_{metric}" for asset in assets for metric in metrics]
    internal_history_db.sort_index(inplace=True)
    return internal_history_db.loc[:, assets_to_filter]


def _iloc_filter_assets(internal_history_db: pd.DataFrame, assets_to_filter: list) -> pd.DataFrame:
    """Filter the in-memory DB by the given asset columns using integer-location based indexing.

    Params:
        internal_history_db: The internal history database DataFrame.
        assets_to_filter:    A list of asset columns to filter by.

    Returns:
        A DataFrame with columns filtered by the given asset columns using iloc.
    """
    assets_to_filter_idx = [internal_history_db.columns.tolist().index(asset) for asset in assets_to_filter]
    return internal_history_db.iloc[:, assets_to_filter_idx]


def _uniform_columns(internal_history_db: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Modify the column names of the internal history database to be uniform.

    If the metrics list contains only one item, the column names are
    set to the base asset symbol without the metric.

    Params:
        internal_history_db: The internal history database DataFrame.
        metrics:             A list of metrics.

    Returns:
        A DataFrame with modified column names for uniformity.
    """
    if len(metrics) == 1:
        internal_history_db.columns = pd.Index(get_col_names(internal_history_db.columns))
    return internal_history_db


def _calculate_index(position: str, skip: int) -> int:
    """Calculate the index in the internal history database.

    position == "latest" means most recent candle.

    Params:
        position: The requested position of the data point.
        skip:     The number of candles to skip.
    """
    return (-1 - skip) if position == "latest" else (0 + skip)


class MarketHistory:
    """MarketHistory is the central unit for fetching and storing market history data.

    It keeps a local cache of the candle history data in memory (pd.DataFrame) and on disk (feather file).
    It provides methods for data retrieval, updates, and caching.
    It interacts with various backends via interfaces: Currently MongoDB and PostgreSQL.
    Bitfinex is the only supported exchange at the moment.

    Attributes:
        root (Tradeforce): The main Tradeforce instance.
        config (Config): User defined configurations.
        log (Logger): Logger init with the name of this module.
        internal_history_db (DataFrame): Stores market history data in-memory. Mirror of backend DB.
        is_load_history_finished (Event): Indicates whether the loading of the history from the backend is finished.
        path_current (Path): The current working directory.
        path_local_cache (Path): The path to the local cache.
        local_cache_filename (str): The filename for the local cache.
    """

    def __init__(self, root: Tradeforce, path_current: str | None = None, path_history_dumps: str | None = None):
        """Initialize a MarketHistory instance.

        Params:
            root:               A reference to the main Tradeforce instance.
            path_current:       The current working directory. If None, uses the working directory from the config.
            path_history_dumps: The directory for history dumps. Defaults to "data" if None.
        """
        self.root = root
        self.config = root.config
        self.log = root.logging.get_logger(__name__)
        self.internal_history_db = pd.DataFrame()
        self.is_load_history_finished = asyncio.Event()

        # Set paths
        self.path_current = Path(path_current) if path_current else self.config.working_dir
        history_dumps_dir = path_history_dumps if path_history_dumps else "data"
        self.path_local_cache = self.path_current / history_dumps_dir
        self.local_cache_filename = f"{self.config.dbms_history_entity_name}.arrow"

    def _create_backend_db_index(self) -> None:
        """Create a unique index on the 't' column of the backend database
        if it's a new history entity.
        """

        if self.root.backend is None:
            return

        if not self.root.backend.is_new_history_entity:
            return

        self.root.backend.is_new_history_entity = False
        self.root.backend.create_index(self.config.dbms_history_entity_name, "t", unique=True)

    def _inmemory_db_set_index(self) -> None:
        """Set the index for the internal_history_db DataFrame
        to the 't' column if it exists.
        """

        if "t" not in self.internal_history_db.columns:
            self.log.debug("Cannot set index for internal history db. Column 't' not found.")
            return

        try:
            self.internal_history_db.set_index("t", inplace=True)
        except (KeyError, ValueError, TypeError, AttributeError):
            pass

    def _postgres_create_history_table(self) -> None:
        """Create the history table in the PostgreSQL database if it does not exist yet.

        This helper method is only used for PostgreSQL, as it requires existing tables and
        columns before the first data insert.
        """
        if self.root.backend is None:
            return

        if not hasattr(self.root.backend, "create_table"):
            return

        if self.root.backend.is_new_history_entity:
            self.root.backend.create_table.history(self.root.asset_symbols)

    def update_internal_db_market_history(self, df_history_update: pd.DataFrame) -> None:
        """Update the internal market history DataFrame with new candle data.

        Params:
            df_history_update: A DataFrame containing new market history data to be added.
        """
        self.internal_history_db = pd.concat([self.internal_history_db, df_history_update], axis=0)
        self._inmemory_db_set_index()
        self.internal_history_db = set_internal_db_column_type(self.internal_history_db)
        self.internal_history_db.sort_index(inplace=True)

    async def load_history(self) -> pd.DataFrame:
        """Load market history data using a specified method or try all available methods.

        Attempts to load market history data from one of the following sources:
        local cache, backend, or API. The source can be forced by the configuration settings.
        If the source is set to "none", it will try all available methods in the mentioned order.
        After successfully loading the history, it runs the post-processing.

        Returns:
            pd.DataFrame: The loaded market history data as a DataFrame.

        Raises:
            SystemExit: If the data could not be loaded using the specified method or any method
                            when trying all available methods.
        """
        load_method = self.config.force_source

        if load_method == "none":
            await self._try_all_load_methods()
        elif load_method == "local_cache":
            self._load_via_local_cache(raise_on_failure=True)
        elif load_method == "backend":
            self._load_via_backend(raise_on_failure=True)
        elif load_method == "api":
            await self._load_via_api(raise_on_failure=True)

        if self.internal_history_db.empty:
            raise SystemExit(
                f"[ERROR] No local_cache or DB storage found for '{self.config.dbms_history_entity_name}'. "
                + f"No API connection ({self.config.exchange}) to fetch new exchange history: "
                + "Set update_mode to 'once' or 'live' or check your internet connection."
            )

        await self._post_process()

        self.log.info(
            "Market history loaded via %s from %s assets from %s",
            self.config.force_source,
            len(self.root.asset_symbols),
            self.config.exchange,
        )

        # Trigger event to indicate that the history has been loaded. Websockets can now start.
        self.is_load_history_finished.set()

        return self.internal_history_db

    async def _try_all_load_methods(self) -> None:
        """Attempt to load market history data using all available methods in sequence.

        Tries to load market history data from the following sources, in order:
        local cache, backend, and API. If none of the methods succeed,
        it raises a SystemExit exception with an error message.

        Raises:
            SystemExit: If the data could not be loaded using any available methods.
        """
        try_next_method = self._load_via_local_cache(raise_on_failure=False)
        if try_next_method:
            try_next_method = self._load_via_backend(raise_on_failure=False)
        if try_next_method:
            try_next_method = await self._load_via_api(raise_on_failure=False)

        if try_next_method:
            raise SystemExit(
                "[ERROR] Was not able to load market history via local or remote db storage nor remote api fetch.\n"
                + "Check your local file paths, DB connection or internet connection."
            )

    def _load_via_local_cache(self, raise_on_failure: bool = False) -> bool:
        """Load market history data from the local cache.

        Attempts to load market history data from a local cache file
        in the feather format. If the data is successfully loaded, it updates the
        internal history database and sets the force_source attribute to "local_cache".

        Params:
            raise_on_failure: If True, raise a SystemExit if unable to load market
                history via API. This flag is set to True if the user
                wants to force the data loading method to be used.

        Returns:
            True if the data could not be loaded and the next method should be tried,
                False otherwise.

        Raises:
            SystemExit: If raise_on_failure is True and data loading fails.
        """
        read_feather_path = Path(self.path_local_cache, self.local_cache_filename)

        try:
            internal_history_db = pd.read_feather(read_feather_path)
        except Exception:
            self.log.info("No local_cache of market history found at: %s", read_feather_path)
            if raise_on_failure:
                raise SystemExit("[ERROR] Was not able to load market history via local_cache.")
            return True
        else:
            self.update_internal_db_market_history(internal_history_db)
            self.config.force_source = "local_cache"
            return False

    def _load_via_backend(self, raise_on_failure: bool = False) -> bool:
        """Load market history data from the configured backend.

        Attempts to load market history data from the configured
        backend database (MongoDB or PostgreSQL). If the data is successfully
        loaded, it updates the internal history database and sets the force_source
        attribute to "backend". raise_on_failure is a flag that gets set to True
        if the user wants to force the data loading method to be used.

        Params:
            raise_on_failure: If True, raise a SystemExit if unable to load market
                history via API. This flag is set to True if the user
                wants to force the data loading method to be used.

        Returns:
            True if the data could not be loaded and the next method should be tried,
                False otherwise.

        Raises:
            SystemExit: If raise_on_failure is True and data loading fails.
        """
        if not self.root.backend.is_new_history_entity:
            exchange_market_history = self.root.backend.query(self.config.dbms_history_entity_name, sort=[("t", 1)])
            internal_history_db = pd.DataFrame(exchange_market_history)
            self.update_internal_db_market_history(internal_history_db)
        else:
            dbms_name = {"mongodb": "MongoDB Collection", "postgresql": "PostgreSQL Table"}.get(self.config.dbms)
            self.log.info("%s '%s' does not exist!", dbms_name, self.config.dbms_history_entity_name)
            if raise_on_failure:
                raise SystemExit(f"[ERROR] Was not able to load market history via {self.config.dbms}.")
            return True

        if not self.internal_history_db.empty:
            self.config.force_source = "backend"
            return False

        return True

    # TODO: Check if the old logic can be removed

    # if not self.root.asset_symbols:
    #     end = await self.root.exchange_api.get_latest_remote_candle_timestamp()
    #     start = get_time_minus_delta(end, delta=f"{self.config.fetch_init_timeframe_days}days")["timestamp"]
    #     self._postgres_create_history_table()
    # else:

    async def _fetch_history_start_end_times(self) -> tuple[int, int]:
        """Determine the start and end timestamps for fetching market history data.

        First fetch the market history and update it. If there are no asset symbols
        yet, fetch the latest remote candle timestamp and calculate the start timestamp by
        subtracting the configured initial timeframe.

        If asset symbols exist, the function determines the start and end timestamps based on the
        local candle timestamps and the configured initial timeframe.

        Returns:
            A tuple containing the start and end timestamps for fetching market history data.
        """
        await self._fetch_market_history_and_update()
        latest_local_candle_timestamp = self.get_local_candle_timestamp(position="latest")
        start_time = get_time_minus_delta(
            latest_local_candle_timestamp, delta=f"{self.config.fetch_init_timeframe_days}days"
        )
        start = start_time["timestamp"]

        first_local_candle_timestamp = self.get_local_candle_timestamp(position="first")
        end_time = get_time_minus_delta(first_local_candle_timestamp, delta=self.config.candle_interval)
        end = end_time["timestamp"]

        return start, end

    async def _load_via_api(self, raise_on_failure: bool = False) -> bool:
        """Load market history via API from the configured exchange.

        If no public API is available, returns True,
        indicating that the next method in the loading
        chain should be tried.

        Fetch the start and end timestamps for market history
        data and update the internal history database with the
        data fetched between these timestamps.

        Params:
            raise_on_failure: If True, raise a SystemExit if
                                unable to load market history via API.
                                This flag is set to True if the user wants to
                                force the data loading method to be used.

        Returns:
            False if the internal history database is not empty
            after loading data via API, otherwise True.
            (False is returned to indicate that the next
            method in the loading chain should be tried).

        Raises:
            SystemExit: If raise_on_failure is True and data loading fails.
        """
        if not self.root.exchange_api.api_pub:
            return True

        self.log.info("Fetching market history via API from %s.", self.config.exchange)
        start, end = await self._fetch_history_start_end_times()

        self.log.info(
            "Fetching %s (%s - %s) of market history from %s assets",
            f"{self.config.fetch_init_timeframe_days} days",
            start,
            end,
            len(self.root.asset_symbols),
        )

        if start < end:
            await self.update(start=start, end=end)

        if not self.internal_history_db.empty:
            self.config.force_source = "api"
            return False

        if raise_on_failure:
            raise SystemExit("[ERROR] Was not able to load market history via API.")

        return True

    async def _fetch_market_history_and_update(self) -> None:
        """Fetch market history data for relevant assets

        and update the internal history database: Retrieve relevant assets and
        their data based on the initial criteria. Filters the assets data to obtain
        the required metrics (open, high, low, close, volume). Then, creates a
        history table in the PostgreSQL database if it doesn't exist, and updates
        the internal history database with the fetched market history data.
        """
        relevant_assets = await self.get_init_relevant_assets()
        filtered_assets = [
            f"{asset}_{metric}" for asset in relevant_assets["assets"] for metric in ["o", "h", "l", "c", "v"]
        ]
        history_data = relevant_assets["data"][filtered_assets]

        self._postgres_create_history_table()

        await self.update(history_data=history_data)

    async def _post_process(self) -> None:
        """Perform post-processing steps after loading market history data.

        Sets the internal history database's column types (float),
        saves the data to the local_cache if enabled, retrieves and updates
        asset_symbols if needed, creates the history table in the backend
        database, and performs database synchronization checks.

        If the update mode is set to "once", it updates the market history
        if the latest local candle timestamp is earlier than the latest
        remote candle timestamp.

        Additionally, if check_db_consistency flag is set to True,
        it performs a consistency check on the database.
        """
        self.internal_history_db = set_internal_db_column_type(self.internal_history_db)

        if self.config.force_source in ("api", "backend", "none"):
            if self.config.local_cache:
                self.save_to_local_cache()

        if not self.root.asset_symbols:
            self.get_asset_symbols(updated=True)

        self._postgres_create_history_table()

        # Postgres does not need an index, as it is created automatically on the primary key
        if not self.config.dbms == "postgresql":
            self._create_backend_db_index()

        if self.config.check_db_sync:
            self.root.backend.db_sync_check()

        if self.config.update_mode == "once":
            start = self.get_local_candle_timestamp(position="latest", offset=1)
            end = await self.root.exchange_api.get_latest_remote_candle_timestamp()
            if start < end:
                await self.update(start=start, end=end)
            else:
                self.log.info("Market history is already UpToDate (%s)", start)

        if self.config.check_db_consistency is True:
            self.root.backend.check_db_consistency()

    def save_to_local_cache(self) -> None:
        """Save the internal history database to local cache as a Feather file.

        Create the cache directory if it does not exist, and save
        the internal history database as a Feather file in the local cache directory.
        """

        Path(self.path_local_cache).mkdir(parents=True, exist_ok=True)
        self.internal_history_db.reset_index(drop=False).to_feather(self.path_local_cache / self.local_cache_filename)
        self.log.info("Assets dumped via local_cache to: %s", str(self.path_local_cache / self.local_cache_filename))

    def get_market_history(
        self,
        assets: list[str] | None = None,
        start: int | None = None,
        end: int | None = None,
        from_list: np.ndarray | None = None,
        latest_candle: bool = False,
        idx_type: str = "loc",
        metrics: list | None = None,
        pct_change: bool = False,
        pct_as_factor: bool = True,
        pct_first_row: int | float | None = None,
        fill_na: bool = False,
        uniform_cols: bool = False,
    ) -> pd.DataFrame:
        """Retrieve (a filtered subset of) the market history data as a DataFrame.

            Params:
                assets: A list of asset symbols to include in the result.
                            Defaults to all available assets.

                start: The start index or timestamp for the data.

                end: The end index or timestamp for the data.

                from_list: A NumPy array of indexes or timestamps to select from.

                latest_candle: If True, return only the latest candle.

                idx_type: Indexing type to use. Either "loc" (default) or "iloc".

                metrics: A list of metrics to include in the result.
                            Defaults to ["o", "h", "l", "c", "v"].

                pct_change: If True, return percentage change values.

                pct_as_factor: If True, return percentage change
                                as a factor (1.0 + pct_change).

                pct_first_row: The first row value to use
                                when calculating percentage change.

                fill_na: If True, fill NaN values in the DataFrame.

                uniform_cols: If True, return a DataFrame with uniform column naming.

        Returns:
            A DataFrame containing the filtered market history data.
        """

        metrics = metrics if metrics else ["o", "h", "l", "c", "v"]
        assets = assets if assets else self.get_asset_symbols(updated=True)

        if latest_candle:
            idx_type = "iloc"
            start = -1

        if from_list is not None:
            idx_type = "None"

        if fill_na and not self.root.backend.is_filled_na:
            _df_fill_na(self.internal_history_db)
            self.root.backend.is_filled_na = True

        if idx_type == "loc":
            internal_history_db = _filter_assets(self.internal_history_db, assets, metrics).loc[start:end]
        elif idx_type == "iloc":
            assets_to_filter = [f"{asset}_{metric}" for asset in assets for metric in metrics]
            internal_history_db = _iloc_filter_assets(self.internal_history_db, assets_to_filter).iloc[start:end]
        elif from_list is not None:
            internal_history_db = self.internal_history_db.loc[from_list]
        else:
            raise ValueError("Invalid value for idx_type")

        if pct_change:
            internal_history_db = _get_pct_change(internal_history_db, pct_first_row, as_factor=pct_as_factor)

        if uniform_cols:
            internal_history_db = _uniform_columns(internal_history_db, metrics)

        return internal_history_db

    async def update(self, history_data: pd.DataFrame | None = None, start: int = -1, end: int = -1) -> None:
        """Update the market history with new data, either provided or fetched from the API.

        If history_data is provided, update the backend database with the given data.
        Otherwise, fetch the market history data within the specified start and end
        timestamp range, using the configured candle interval, and update the backend
        database accordingly.

        Params:
            history_data: A DataFrame containing market history data to update.
                        If None, data will be fetched from the API.
            start: The start timestamp for the data to fetch.
                Only used if history_data is None. Defaults to -1.
            end: The end timestamp for the data to fetch.
                Only used if history_data is None. Defaults to -1.
        """
        if history_data is not None:
            df_history_update = history_data
        else:
            timestamp_intervals = _get_timestamp_intervals(start, end, self.config.candle_interval)
            df_history_update_list = []
            for interval in timestamp_intervals:
                self.log.info("Fetching history interval: %s", interval)
                i_start = interval[0]
                i_end = interval[1]
                df_history_update_interval = await self.root.market_updater_api.update_market_history(
                    start=i_start, end=i_end
                )
                if df_history_update_interval is None:
                    continue
                df_history_update_list.append(df_history_update_interval)
            df_history_update = pd.concat(df_history_update_list, axis=0)
            df_history_update = df_history_update.iloc[~df_history_update.index.duplicated()]
        self.root.backend.db_add_history(df_history_update)

    # --------#
    # Getters #
    # --------#

    def get_asset_symbols(self, updated: bool = False) -> list[str]:
        """Retrieve a list of all asset symbols in the in-memory database.

        Params:
            updated: If True, update the asset symbols list by extracting
                them from the internal history database.

        Returns:
            A list of asset symbols as strings.
        """
        if updated:
            self.root.asset_symbols = get_col_names(self.internal_history_db.columns)
        return self.root.asset_symbols

    async def get_init_relevant_assets(self) -> DictRelevantAssets:
        """Fetch the relevant assets on initialization.

        Fetches the initial relevant assets based on the root configuration and the
        relevant assets cap. Then update the global asset_symbols.

        Returns:
            Dict containing information about the relevant assets.
        """
        relevant_assets = await get_init_relevant_assets(self.root, capped=self.config.relevant_assets_cap)
        self.root.asset_symbols = relevant_assets["assets"]
        return relevant_assets

    def get_local_candle_timestamp(self, position: str = "latest", skip: int = 0, offset: int = 0) -> int:
        """Retrieve the local candle timestamp for a given position, skip, and offset.

        Params:
            position: A string indicating the position of the candle ("latest" or "first").
            skip:     An integer representing the number of candles to skip before retrieving the timestamp.
            offset:   An integer representing the number of candles to offset from the selected position.

        Returns:
            An integer representing the local candle timestamp.
        """
        idx = _calculate_index(position, skip)
        latest_candle_timestamp = self._get_timestamp_from_history(idx)
        candle_interval_in_ms = candle_interval_to_ms(self.config.candle_interval)
        latest_candle_timestamp += offset * candle_interval_in_ms
        return latest_candle_timestamp

    def _get_timestamp_from_history(self, idx: int) -> int:
        """Retrieve the timestamp from the market history database at the specified index.

        If the internal history database is not empty, retrieve the timestamp from the internal
        history database. If the internal history database is empty and the DBMS is MongoDB,
        query the backend to retrieve the timestamp.

        Params:
            idx: Integer representing the index of the desired timestamp.

        Returns:
            Integer representing the timestamp at the specified index.
        """
        if not self.internal_history_db.empty:
            return int(self.internal_history_db.index.values[idx])

        elif self.config.dbms == "mongodb" and not self.root.backend.is_new_history_entity:
            sort_id = -1 if idx < 0 else 1
            query_result = self.root.backend.query(
                self.config.dbms_history_entity_name, sort=[("t", sort_id)], skip=abs(idx), limit=1
            )
            return int(query_result[0]["t"])
        return 0

    def get_history_size(self) -> int:
        """Retrieve the size of the market history database.

        Returns:
            Integer representing the size of the market history database.
        """
        return len(self.internal_history_db)
