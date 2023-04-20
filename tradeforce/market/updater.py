""" market/updater.py

Module: tradeforce.market_updater
---------------------------------

Provides the MarketUpdater class for updating the market data (candles) of assets.
It fetches market history data for assets within a specified timeframe, processes it,
and returns a DataFrame containing the updated market history data.

The module also provides helper functions for handling timestamps, timeframes, asynchronous
fetching of asset symbols, and DataFrame operations.

Classes:
    MarketUpdater: Responsible for updating the market data/candles of assets.

Main Function:
    update_market_history: Update the market history data within the specified timeframe.

"""

from __future__ import annotations

from asyncio import sleep as asyncio_sleep
from tqdm.asyncio import tqdm
from time import process_time

from functools import reduce

from typing import TYPE_CHECKING, AsyncIterator
import pandas as pd

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce.main import Tradeforce

from tradeforce.custom_types import DictTimeframe, DictTimeframeExtended, DictMarketHistoryUpdate
from tradeforce.utils import (
    get_now,
    get_timedelta,
    get_time_minus_delta,
    get_reference_index,
)


def _append_market_history_update(assets_hist_update: dict, assets_candles: list, symbol_current: str) -> list:
    """Update the market history of a given asset with new candle data.

    Append new candle data to the existing market history of an asset.
    If the asset is not present in the assets_hist_update dictionary,
    initialize its history with the new data.

    Params:
        assets_hist_update: Dict containing the market history of assets.
        assets_candles: A list of new candle data for the current asset.
        symbol_current: The asset symbol (e.g. "BTC") for which the
                            market history is being updated.

    Returns:
        A list containing the updated market history of the current asset.
    """
    current_asset = assets_hist_update.get(symbol_current, [])
    current_asset.extend(
        [
            {
                "t": i[0],
                symbol_current + "_o": i[1],
                symbol_current + "_c": i[2],
                symbol_current + "_h": i[3],
                symbol_current + "_l": i[4],
                symbol_current + "_v": i[5],
            }
            for i in assets_candles
        ]
    )
    return current_asset


def _calculate_ms_until_wait_over(start_timestamp: int, end_timestamp: int) -> int:
    """Calculate the number of milliseconds to wait before the specified time window is over.

    This wait time is added to not request data that is not yet available on the exchange.
    The function computes the difference between the end_timestamp and the start_timestamp
    plus an additional waiting time (e.g., 2 minutes).

    Params:
        start_timestamp: The starting timestamp of the time window.
        end_timestamp: The ending timestamp of the time window.

    Returns:
        The number of milliseconds to wait before the specified time window is over.
    """
    ms_additional_wait = get_timedelta("2min")["timestamp"]
    return end_timestamp - (start_timestamp + ms_additional_wait)


def _get_end_timestamp(end: int | None = None) -> int:
    """Determine the end timestamp for a given time window.

    If an end timestamp is provided, return it. Otherwise, return the current timestamp.

    Params:
        end: The optional end timestamp for the time window.

    Returns:
        The end timestamp for the given time window."""
    if end:
        return end

    now = get_now()
    return now["timestamp"]


async def _get_symbols_async(symbols: list[str]) -> AsyncIterator[str]:
    """Asynchronously iterate through a list of asset symbols.

    This function enables processing multiple asset symbols concurrently, which can be useful
    when fetching market data or executing tasks on multiple assets simultaneously.

    Utilizes tqdm to display a progress bar.

    Params:
        symbols: A list of asset symbols (e.g., ["BTC", "ETH", "XRP", "ADA", "LTC"]).

    Yields:
        Asset symbols as strings, one at a time, asynchronously.
    """
    for symbol in tqdm(symbols, desc="Fetching market history"):
        yield symbol


class MarketUpdater:
    """MarketUpdater is responsible for updating the market data / candles of assets."""

    def __init__(self, root: Tradeforce):
        """Initialize the MarketUpdater class.

        Params:
            root: The instance of the Tradeforce class, which contains configuration and logging.
        """
        self.root = root
        self.config = root.config
        self.log = root.logging.get_logger(__name__)

    def _get_start_timestamp_adjusted(self, start_timestamp: int) -> int:
        """Adjust the start timestamp by adding the candle frequency in milliseconds.

        Params:
            start_timestamp: The original start timestamp.

        Returns:
            The adjusted start timestamp.
        """
        candle_freq_in_ms = get_timedelta(self.config.candle_interval)["timestamp"]
        return start_timestamp + candle_freq_in_ms

    async def _get_start_timestamp(self, start: int | None = None) -> int:
        """Determine the start timestamp for a given time window.

        If a start timestamp is provided, return it. Otherwise, return the latest local candle
        timestamp or the latest remote candle timestamp minus a specified time delta.

        Params:
            start: The optional start timestamp for the time window.

        Returns:
            The start timestamp for the given time window.
        """
        if start:
            return start

        start_timestamp = self.root.market_history.get_local_candle_timestamp(position="latest")
        if start_timestamp == 0:
            start_timestamp = await self.root.exchange_api.get_latest_remote_candle_timestamp(
                minus_delta=f"{self.config.fetch_init_timeframe_days}days"
            )
        else:
            start_timestamp = self._get_start_timestamp_adjusted(start_timestamp)

        return start_timestamp

    async def _get_timeframe(self, start: int | None = None, end: int | None = None) -> DictTimeframeExtended:
        """Determine the timeframe for a given time window.

        Calculate start and end timestamps, and convert them to datetime objects. Additionally,
        compute the time until the waiting period is over and new candle data is available.

        Params:
            start: The optional start timestamp for the time window.
            end: The optional end timestamp for the time window.

        Returns:
            Dict containing the start and end timestamps and datetimes, along with the
            number of milliseconds until the waiting period is over.
        """
        start_timestamp = await self._get_start_timestamp(start)
        start_datetime = pd.to_datetime(start_timestamp, unit="ms", utc=True)

        end_timestamp = _get_end_timestamp(end)
        end_datetime = pd.to_datetime(end_timestamp, unit="ms", utc=True)

        ms_until_wait_over = _calculate_ms_until_wait_over(start_timestamp, end_timestamp)

        timeframe: DictTimeframe = {
            "start_timestamp": start_timestamp,
            "start_datetime": start_datetime,
            "end_timestamp": end_timestamp,
            "end_datetime": end_datetime,
        }

        return {"timeframe": timeframe, "ms_until_wait_over": ms_until_wait_over}

    async def _fetch_candles(self, symbol: str, timeframe: DictTimeframe) -> list:
        """Fetch candles for a given symbol and timeframe.

        Params:
            symbol: The asset symbol, e.g. "BTCUSD".
            timeframe: The dictionary containing start and end timestamps for the time window.

        Returns:
            A list of fetched candle data for the given symbol and timeframe.
        """
        return await self.root.exchange_api.get_public_candles(
            symbol=symbol,
            timestamp_start=timeframe["start_timestamp"],
            timestamp_end=timeframe["end_timestamp"],
        )

    async def _fetch_market_history(self, timeframe: DictTimeframe) -> DictMarketHistoryUpdate:
        """Fetch market history data for all asset symbols within a specified timeframe.

        Fetch the market history data for each asset symbol present
        in the root.asset_symbols list.

        If there are any candle updates for the symbol,
        append the market history updates for that symbol.

        Take API call rate limits into account and add
        an asynchronous sleep to avoid hitting the rate limits.

        Params:
            timeframe: The dictionary containing start and end timestamps for the time window.

        Returns:
            Dict containing the market history update for each asset symbol.
        """
        market_history_update: DictMarketHistoryUpdate = {}

        async for symbol in _get_symbols_async(self.root.asset_symbols):
            time_start = process_time()
            candle_update = await self._fetch_candles(symbol, timeframe)
            if len(candle_update) > 0:
                market_history_update_for_symbol = _append_market_history_update(
                    market_history_update, candle_update, symbol
                )
                market_history_update[symbol] = market_history_update_for_symbol

            time_elapsed_difference = 1 - (process_time() - time_start)
            if time_elapsed_difference > 0:
                await asyncio_sleep(time_elapsed_difference)

        return market_history_update

    def _create_asset_history_df(self, asset_hist: list) -> pd.DataFrame:
        """Create a DataFrame given the asset history data.

        Params:
            asset_hist: A list of asset history data.

        Returns:
            A sorted pandas DataFrame containing asset history data.
        """
        df_hist = pd.DataFrame(asset_hist)
        df_hist.sort_values("t", inplace=True, ascending=True)
        return df_hist

    def _merge_dataframes(self, history_df_list: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge a list of DataFrames containing asset history data.

        Params:
            history_df_list: A list of pandas DataFrames containing asset history data.

        Returns:
            A merged pandas DataFrame containing asset history data from all DataFrames in the list.
        """
        return reduce(
            lambda df_left, df_right: pd.merge_asof(df_left, df_right, on="t", direction="nearest", tolerance=60000),
            history_df_list,
        )

    def _convert_market_history_to_df(
        self, history_update: DictMarketHistoryUpdate, timeframe: dict[str, pd.Timestamp]
    ) -> pd.DataFrame:
        """Convert the fetched market history update data into a DataFrame.

        Create a DataFrame for each asset symbol and append them to a list.
        If a timeframe is provided, also append a reference index DataFrame to the list.
        Finally, merge all the DataFrames in the list using the _merge_dataframes method
        and set the index to the timestamp column.

        Params:
            history_update: Dict containing the market history update for each asset symbol.
            timeframe: Dict containing start and end timestamps as Timestamp objects.

        Returns:
            DataFrame containing the merged market history data for all asset symbols.

        Raises:
            A warning is logged if there is no market update data available to create a DataFrame.
        """
        if not history_update:
            self.log.warning("No market update, cannot create dataframe!")
            return pd.DataFrame()

        history_df_list = []

        if timeframe is not None:
            history_df_list.append(get_reference_index(timeframe, freq=self.config.candle_interval))

        for asset_hist in history_update.values():
            if len(asset_hist) > 0:
                history_df_list.append(self._create_asset_history_df(asset_hist))

        internal_history_db_update = self._merge_dataframes(history_df_list)
        internal_history_db_update.set_index("t", inplace=True)

        return internal_history_db_update

    async def _update_market_history_if_needed(self, timeframe: DictTimeframeExtended) -> DictMarketHistoryUpdate:
        """Fetch market history updates if the wait time is over.

        Check if the required wait time between updates is over
        and fetch the market history data if it is.

        Params:
            timeframe: Dict containing the start and end
                        timestamps, and the remaining milliseconds
                        until the wait time is over.

        Returns:
            Dict containing the market history update for each asset symbol.
            An empty dictionary if the wait time is not over.

        Raises:
            A warning is logged if there is insufficient market update data fetched.
            An info message is logged if the update request is too early.
        """
        if timeframe["ms_until_wait_over"] > 0:
            market_history_update = await self._fetch_market_history(timeframe["timeframe"])
            if len(market_history_update) <= 1:
                self.log.warning(
                    "No market update since %s. Maybe exchange down?",
                    str(pd.to_timedelta(timeframe["ms_until_wait_over"], unit="ms")),
                )
            return market_history_update
        else:
            self.log.info(
                "Update request too early. Frequency is %s. Wait at least %s min for next try!",
                self.config.candle_interval,
                int(abs(timeframe["ms_until_wait_over"]) / 1000 // 60),
            )
            return {}

    async def _initialize_start_timestamp(self, init_timespan: str) -> int:
        latest_remote_candle_timestamp = await self.root.exchange_api.get_latest_remote_candle_timestamp()
        return get_time_minus_delta(timestamp=latest_remote_candle_timestamp, delta=init_timespan)["timestamp"]

    async def update_market_history(
        self, start: int | None = None, end: int | None = None, init_timespan: str | None = None
    ) -> pd.DataFrame:
        """Update the market history data within the specified timeframe.

        Update the market history by fetching new data, if needed, based on the
        given start and end timestamps, or the init_timespan provided.

        Params:
            start: The start timestamp for the time window (optional).
            end: The end timestamp for the time window (optional).
            init_timespan: A string representing the initial timespan,
                            e.g., "30min" (optional).

        Returns:
            A DataFrame containing the updated market history data.

        Notes:
            If both start and end timestamps are provided,
                they will be used as the time window.

            If init_timespan is provided,
                it will be used to determine the start timestamp.

            If neither start nor end timestamps are provided,
                the method will use the latest available data.
        """
        if init_timespan:
            start = await self._initialize_start_timestamp(init_timespan)

        if not self.root.asset_symbols:
            self.root.asset_symbols = await self.root.exchange_api.get_active_assets()

        timeframe = await self._get_timeframe(start=start, end=end)

        market_history_update = await self._update_market_history_if_needed(timeframe)

        if len(market_history_update) > 1:
            timeframe_datetime = {
                "start_datetime": timeframe["timeframe"]["start_datetime"],
                "end_datetime": timeframe["timeframe"]["end_datetime"],
            }

            internal_history_db_update = self._convert_market_history_to_df(
                market_history_update, timeframe=timeframe_datetime
            )
        else:
            internal_history_db_update = pd.DataFrame()

        return internal_history_db_update
