"""_summary_
Returns:
    _type_: _description_
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING
from tradeforce.custom_types import DictRelevantAssets
from tradeforce.utils import ms_to_ns, get_col_names, ns_to_ms, ns_to_ms_array, get_time_minus_delta
from tradeforce.market.metrics import get_init_relevant_assets

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce import Tradeforce


def get_pct_change(df_history: pd.DataFrame, pct_first_row: int, as_factor: bool = True) -> pd.DataFrame:
    """Calculates the percentage change between each row and the previous row of DataFrame df_history,
    and optionally converts the changes to factors by adding 1 to each value. e.g. -0.1 -> 0.9
    Allows setting a specific value for the first row of the output DataFrame.
    Also renames the columns of the output DataFrame to include "_pct". e.g. "BTC_o" -> "BTC_o_pct"
    """
    df_history_pct = df_history.pct_change()
    if as_factor:
        df_history_pct += 1
    if pct_first_row is not None:
        df_history_pct.iloc[0] = pct_first_row
    df_history_pct.columns = pd.Index([f"{col}_pct" for col in df_history_pct.columns])
    return df_history_pct


# TODO: make freq dynamic based on min time delta between candles
def get_timestamp_intervals(start: int, end: int) -> list[tuple[int, int]]:
    timestamp_intervals = [(-1, -1)]
    if start != -1 and end != -1:
        # 50000min == 10000 * 5min -> 10000 max len @ bitfinex api
        timestamp_array = ns_to_ms_array(pd.date_range(ms_to_ns(start), ms_to_ns(end), tz="UTC", freq="50000min").asi8)
        timestamp_list = np.array(timestamp_array).tolist()
        if timestamp_list[-1] != end:
            timestamp_list.append(end)
        timestamp_intervals = list(zip(timestamp_list, timestamp_list[1:]))
    return timestamp_intervals


def set_internal_db_column_type(inmemory_db: pd.DataFrame) -> pd.DataFrame:
    return inmemory_db.astype(np.float32)


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
    """
    MarketHistory is the central unit for fetching and storing market history data.
    It keeps a local cache of the candle history data in memory (pd.DataFrame) and on disk (feather file).
    It provides methods for data retrieval, updates, and caching.
    It interacts with various backends via interfaces: Currently MongoDB and PostgreSQL.
    Bitfinex is the only supported exchange at the moment.

    Attributes:
        root (Tradeforce): A reference to the main Tradeforce instance.
        config (Config): A reference to the configuration object.
        log (Logger): A reference to the logger object.
        internal_history_db (pd.DataFrame): Stores market history data in memory.
        path_current (Path): The current working directory.
        path_local_cache (Path): The path to the local cache.
        local_cache_filename (str): The filename for the local cache.
    """

    def __init__(self, root: Tradeforce, path_current: str | None = None, path_history_dumps: str | None = None):
        self.root = root
        self.config = root.config
        self.log = root.logging.get_logger(__name__)
        self.internal_history_db = pd.DataFrame()

        # Set paths
        self.path_current = Path(path_current) if path_current else self.config.working_dir
        history_dumps_dir = path_history_dumps if path_history_dumps else "data"
        self.path_local_cache = self.path_current / history_dumps_dir
        self.local_cache_filename = f"{self.config.dbms_history_entity_name}.arrow"

    def _create_backend_db_index(self) -> None:
        if not self.root.backend.is_new_history_entity:
            return

        self.root.backend.is_new_history_entity = False
        self.root.backend.create_index(self.config.dbms_history_entity_name, "t", unique=True)

    def _inmemory_db_set_index(self) -> None:
        if "t" not in self.internal_history_db.columns:
            self.log.debug("Cannot set index for internal history db. Column 't' not found.")
            return

        try:
            self.internal_history_db.set_index("t", inplace=True)
        except (KeyError, ValueError, TypeError, AttributeError):
            pass

    def _postgres_create_history_table(self) -> None:
        """Creates the history table in the database if it does not exist yet.

        This helper method is only used for PostgreSQL
        as it needs existing table / columns before the first insert.
        """
        if self.config.dbms == "postgresql" and self.root.backend.is_new_history_entity:
            self.root.backend.create_table.history(self.root.assets_list_symbols)

    def update_internal_db_market_history(self, df_history_update: pd.DataFrame) -> None:
        self.internal_history_db = pd.concat([self.internal_history_db, df_history_update], axis=0)
        # TODO: Check if _inmemory_db_set_index is necessary here
        self._inmemory_db_set_index()
        self.internal_history_db = set_internal_db_column_type(self.internal_history_db)
        self.internal_history_db.sort_index(inplace=True)

    async def load_history(self) -> pd.DataFrame:
        load_method = self.config.force_source

        if load_method == "none":
            await self._try_all_load_methods()
        elif load_method == "local_cache":
            self._load_via_local_cache(raise_on_failure=True)
        elif load_method in ("mongodb", "postgresql"):
            self._load_via_backend(raise_on_failure=True)
        elif load_method == "api":
            await self._load_via_api(raise_on_failure=True)

        if not self.internal_history_db.empty:
            await self.post_process()
            amount_assets = len(self.root.assets_list_symbols)

            self.log.info(
                "Market history loaded via %s from %s assets from %s",
                self.config.force_source,
                amount_assets,
                self.config.exchange,
            )
        else:
            raise SystemExit(f"[ERROR] Was not able to load market history via {load_method}.")

        return self.internal_history_db

    async def _try_all_load_methods(self) -> None:
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

    def _load_via_local_cache(self, raise_on_failure=False) -> bool:
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
            self.config.force_source = self.config.dbms
            return False
        return True

    async def _load_via_api(self, raise_on_failure: bool = False) -> bool:
        if not self.root.exchange_api.bfx_api_pub:
            raise SystemExit(
                f"[ERROR] No local_cache or DB storage found for '{self.config.dbms_history_entity_name}'. "
                + f"No API connection ({self.config.exchange}) to fetch new exchange history: "
                + "Set update_mode to 'once' or 'live' or check your internet connection."
            )
        self.log.info("Fetching market history via API from %s.", self.config.exchange)

        if self.root.assets_list_symbols is not None:
            end = await self.root.exchange_api.get_latest_remote_candle_timestamp()
            start = get_time_minus_delta(end, delta=self.config.history_timeframe)["timestamp"]

            self._postgres_create_history_table()
        else:
            await self._fetch_market_history_and_update()

            latest_local_candle_timestamp = self.get_local_candle_timestamp(position="latest")
            start_time = get_time_minus_delta(latest_local_candle_timestamp, delta=self.config.history_timeframe)
            start = start_time["timestamp"]

            first_local_candle_timestamp = self.get_local_candle_timestamp(position="first")
            end_time = get_time_minus_delta(first_local_candle_timestamp, delta=self.config.candle_interval)
            end = end_time["timestamp"]

        self.log.info(
            "Fetching %s (%s - %s) of market history from %s assets",
            self.config.history_timeframe,
            start,
            end,
            len(self.root.assets_list_symbols),
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
        relevant_assets = await self.get_init_relevant_assets()
        filtered_assets = [
            f"{asset}_{metric}" for asset in relevant_assets["assets"] for metric in ["o", "h", "l", "c", "v"]
        ]
        history_data = relevant_assets["data"][filtered_assets]

        self._postgres_create_history_table()

        await self.update(history_data=history_data)

    async def post_process(self) -> None:
        self.internal_history_db = set_internal_db_column_type(self.internal_history_db)

        if self.config.force_source in ("api", "mongodb", "postgresql", "none"):
            if self.config.local_cache:
                self.save_to_local_cache()

        if self.root.assets_list_symbols is None:
            self.get_asset_symbols(updated=True)

        self._postgres_create_history_table()

        # Postgres does not need an index, as it is created automatically on the primary key
        if not self.config.dbms == "postgresql":
            self._create_backend_db_index()

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
        Path(self.path_local_cache).mkdir(parents=True, exist_ok=True)
        self.internal_history_db.reset_index(drop=False).to_feather(self.path_local_cache / self.local_cache_filename)
        self.log.info("Assets dumped via local_cache to: %s", str(self.path_local_cache / self.local_cache_filename))

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
        pct_first_row=None,
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
                raise SystemExit(
                    f"ERROR: {start} is not a valid pandas time unit abbreviation!\n"
                    + "For reference check: "
                    + "https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases"
                ) from exc
            else:
                start = ns_to_ms(start)

        # Contruct list of columns to filter
        assets_to_filter = [f"{asset}_{metric}" for asset in assets for metric in metrics]
        self.internal_history_db.sort_index(inplace=True)
        if fill_na and not self.root.backend.is_filled_na:
            df_fill_na(self.internal_history_db)
            self.root.backend.is_filled_na = True

        if idx_type == "loc":
            internal_history_db = self.internal_history_db.loc[start:end, assets_to_filter]
        if idx_type == "iloc":
            assets_to_filter_idx = [
                self.internal_history_db.columns.tolist().index(asset) for asset in assets_to_filter
            ]
            internal_history_db = self.internal_history_db.iloc[start:end, assets_to_filter_idx]
        if from_list is not None:
            internal_history_db = self.internal_history_db.loc[from_list]

        if pct_change:
            internal_history_db = get_pct_change(internal_history_db, pct_first_row, as_factor=pct_as_factor)

        if uniform_cols and len(metrics) == 1:
            internal_history_db.columns = get_col_names(internal_history_db.columns)
        return internal_history_db

    async def update(self, history_data: pd.DataFrame | None = None, start=-1, end=-1):
        if history_data is not None:
            df_history_update = history_data
        else:
            timestamp_intervals = get_timestamp_intervals(start, end)
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

    ###########
    # Getters #
    ###########
    def get_asset_symbols(self, updated=False):
        """Returns a list of all asset symbols

        Returns:
            list: str
        """
        if updated:
            self.root.assets_list_symbols = get_col_names(self.internal_history_db.columns)
        return self.root.assets_list_symbols

    async def get_init_relevant_assets(self, capped=None) -> DictRelevantAssets:
        if capped is None:
            capped = self.config.relevant_assets_cap
        relevant_assets = await get_init_relevant_assets(self.root, capped=capped)
        self.root.assets_list_symbols = relevant_assets["assets"]
        return relevant_assets

    def get_local_candle_timestamp(self, position="latest", skip=0, offset=0):
        idx = (-1 - skip) if position == "latest" else (0 + skip)
        latest_candle_timestamp = 0
        if not self.internal_history_db.empty:
            latest_candle_timestamp = int(self.internal_history_db.index[idx])
        elif self.config.dbms == "mongodb" and not self.root.backend.is_new_history_entity:
            sort_id = -1 if position == "latest" else 1
            latest_candle_timestamp = int(
                self.root.backend.query(
                    self.config.dbms_history_entity_name, sort=[("t", sort_id)], skip=skip, limit=1
                )[0]["t"]
            )

        latest_candle_timestamp += offset * 300000
        return latest_candle_timestamp
