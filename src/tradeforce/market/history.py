"""_summary_
Returns:
    _type_: _description_
"""
import sys
import os
from pathlib import Path
import pandas as pd
from tradeforce.utils import ms_to_ns, get_col_names, ns_to_ms, get_time_minus_delta
from tradeforce.market.metrics import get_init_relevant_assets


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

    def __init__(self, root, path_current=None, path_history_dumps=None):

        self.root = root
        self.config = root.config
        self.log = root.logging.getLogger(__name__)
        self.df_market_history = pd.DataFrame()

        # Set paths
        self.path_current = (
            Path(os.path.dirname(os.path.abspath(__file__))) if path_current is None else Path(path_current)
        )
        self.path_local_cache = (
            self.path_current / "data/inputs" if path_history_dumps is None else self.path_current / path_history_dumps
        )
        Path(self.path_local_cache).mkdir(parents=True, exist_ok=True)
        self.local_cache_filename = (
            f"{self.config.exchange}_{self.config.base_currency}_{self.config.dbms_table_or_coll_name}.arrow"
        )

    async def load_history(self) -> pd.DataFrame:
        load_method = self.config.force_source
        try_next_method = True if load_method == "none" else False

        if try_next_method or load_method == "local_cache":
            try_next_method = self.load_via_local_cache(try_next_method)
        if try_next_method or load_method == "mongodb":
            try_next_method = self.load_via_backend(try_next_method)
        if try_next_method or load_method == "api":
            try_next_method = await self.load_via_api(try_next_method)

        if not self.df_market_history.empty:
            await self.post_process()
            amount_assets = len(self.root.assets_list_symbols)

            self.log.info(
                "Market history from %s assets [%s] loaded via %s",
                amount_assets,
                self.config.exchange,
                self.config.force_source,
            )
        else:
            if load_method == "none":
                sys.exit(
                    "[ERROR] Was not able to load market history via local or remote db storage nor remote api fetch.\n"
                    + "Check your local file paths, DB connection or internet connection."
                )
            else:
                sys.exit(f"[ERROR] Was not able to load market history via {load_method}.")
        return self.df_market_history

    def load_via_local_cache(self, try_next_method: bool) -> bool:
        try:
            self.df_market_history = pd.read_feather(self.path_local_cache / self.local_cache_filename)
        except Exception:
            self.log.info(
                "No local_cache of market history found at: %s", self.path_local_cache / self.local_cache_filename
            )
        else:
            self.df_market_history.set_index("t", inplace=True)
            self.df_market_history.sort_index(inplace=True)
            self.config.force_source = "local_cache"
            try_next_method = False
        return try_next_method

    def load_via_backend(self, try_next_method: bool) -> bool:
        if not self.root.backend.is_collection_new:
            cursor = self.root.backend.mongo_exchange_coll.find({}, sort=[("t", 1)], projection={"_id": False})
            self.df_market_history = pd.DataFrame(list(cursor))
        else:
            self.log.info("MongoDB collection '%s' does not exist!", self.config.dbms_table_or_coll_name)
        if not self.df_market_history.empty:
            self.config.force_source = "mongodb"
            try_next_method = False
        return try_next_method

    async def load_via_api(self, try_next_method: bool) -> bool:
        self.log.info("Fetching market history via API from %s.", self.config.exchange)
        if self.root.assets_list_symbols is not None:
            end = await self.root.exchange_api.get_latest_remote_candle_timestamp()
            start = get_time_minus_delta(end, delta=self.config.history_timeframe)["timestamp"]
        else:
            relevant_assets = await get_init_relevant_assets(self.root, capped=self.config.relevant_assets_cap)
            self.root.assets_list_symbols = relevant_assets["assets"]
            filtered_assets = [
                f"{asset}_{metric}" for asset in relevant_assets["assets"] for metric in ["o", "h", "l", "c", "v"]
            ]
            await self.update(history_data=relevant_assets["data"][filtered_assets])
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
        if not self.df_market_history.empty:
            try_next_method = False
            self.config.force_source = "API"
        return try_next_method

    async def post_process(self) -> None:
        try:
            self.df_market_history.set_index("t", inplace=True)
        except (KeyError, ValueError, TypeError, AttributeError):
            pass

        if self.config.force_source in ("api", "mongodb"):
            if self.config.local_cache is True:
                self.save_to_local_cache()

        if self.root.assets_list_symbols is None:
            self.get_asset_symbols(updated=True)

        if self.root.backend.sync_check_needed:
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

    def save_to_local_cache(self):
        self.df_market_history.reset_index(drop=False).to_feather(self.path_local_cache / self.local_cache_filename)
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
        if fill_na and not self.root.backend.is_filled_na:
            df_fill_na(self.df_market_history)
            self.root.backend.is_filled_na = True

        if idx_type == "loc":
            df_market_history = self.df_market_history.loc[start:end, assets_to_filter]
        if idx_type == "iloc":
            assets_to_filter_idx = [self.df_market_history.columns.tolist().index(asset) for asset in assets_to_filter]
            df_market_history = self.df_market_history.iloc[start:end, assets_to_filter_idx]
        if from_list is not None:
            df_market_history = self.df_market_history.loc[from_list]

        if pct_change:
            df_market_history = get_pct_change(df_market_history, pct_first_row, as_factor=pct_as_factor)

        if uniform_cols and len(metrics) == 1:
            df_market_history.columns = get_col_names(df_market_history.columns)
        return df_market_history

    async def update(self, history_data=None, start=-1, end=-1):
        assets_status = None
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
                    return assets_status
                df_history_update_list.append(df_history_update_interval)
            df_history_update = pd.concat(df_history_update_list, axis=0)
            df_history_update = df_history_update.iloc[~df_history_update.index.duplicated()]
        self.root.backend.db_add_history(df_history_update)
        return assets_status

    ###########
    # Getters #
    ###########
    def get_asset_symbols(self, updated=False):
        """Returns a list of all asset symbols

        Returns:
            list: str
        """
        if updated:
            self.root.assets_list_symbols = get_col_names(self.df_market_history.columns)
        return self.root.assets_list_symbols

    def get_local_candle_timestamp(self, position="latest", skip=0, offset=0):
        idx = (-1 - skip) if position == "latest" else (0 + skip)
        latest_candle_timestamp = 0
        if not self.df_market_history.empty:
            latest_candle_timestamp = int(self.df_market_history.index[idx])
        elif self.config.dbms == "mongodb" and not self.root.backend.is_collection_new:
            sort_id = -1 if position == "latest" else 1
            cursor = self.root.backend.mongo_exchange_coll.find({}, sort=[("t", sort_id)], skip=skip, limit=1)
            latest_candle_timestamp = int(cursor[0]["t"])

        latest_candle_timestamp += offset * 300000
        return latest_candle_timestamp
