""" /src/tradeforce/market/updater.py
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
    from tradeforce.main import TradingEngine

from tradeforce.custom_types import DictTimeframe, DictTimeframeExtended, DictMarketHistoryUpdate
from tradeforce.utils import (
    get_now,
    get_timedelta,
    get_time_minus_delta,
    get_reference_index,
)


def append_market_history_update(assets_hist_update: dict, assets_candles: list, symbol_current: str):
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


def calculate_ms_until_wait_over(start_timestamp: int, end_timestamp: int) -> int:
    ms_additional_wait = get_timedelta("2min")["timestamp"]
    return end_timestamp - (start_timestamp + ms_additional_wait)


async def get_start_timestamp(self: MarketUpdater, start: int | None = None) -> int:
    if start:
        return start

    start_timestamp = self.root.market_history.get_local_candle_timestamp(position="latest")
    if start_timestamp == 0:
        start_timestamp = await self.root.exchange_api.get_latest_remote_candle_timestamp(
            minus_delta=self.root.market_history.history_timeframe
        )
    else:
        candle_freq_in_ms = get_timedelta(self.config.candle_interval)["timestamp"]
        start_timestamp += candle_freq_in_ms

    return start_timestamp


def get_end_timestamp(end: int | None = None) -> int:
    if end:
        return end

    now = get_now()
    return now["timestamp"]


class MarketUpdater:
    def __init__(self, root: TradingEngine):
        self.root = root
        self.config = root.config
        self.log = root.logging.get_logger(__name__)

    async def get_timeframe(self, start: int | None = None, end: int | None = None) -> DictTimeframeExtended:
        start_timestamp = await get_start_timestamp(self, start)
        start_datetime = pd.to_datetime(start_timestamp, unit="ms", utc=True)

        end_timestamp = get_end_timestamp(end)
        end_datetime = pd.to_datetime(end_timestamp, unit="ms", utc=True)

        ms_until_wait_over = calculate_ms_until_wait_over(start_timestamp, end_timestamp)

        timeframe: DictTimeframe = {
            "start_timestamp": start_timestamp,
            "start_datetime": start_datetime,
            "end_timestamp": end_timestamp,
            "end_datetime": end_datetime,
        }

        return {"timeframe": timeframe, "ms_until_wait_over": ms_until_wait_over}

    async def _fetch_candles(self, symbol: str, timeframe: dict) -> list:
        return await self.root.exchange_api.get_public_candles(
            symbol=symbol,
            timestamp_start=timeframe["start_timestamp"],
            timestamp_end=timeframe["end_timestamp"],
        )

    async def get_symbols_async(self, symbols: list[str]) -> AsyncIterator[str]:
        for symbol in tqdm(symbols, desc="Fetching market history"):
            yield symbol

    async def fetch_market_history(self, timeframe: dict) -> DictMarketHistoryUpdate:
        market_history_update: DictMarketHistoryUpdate = {}

        async for symbol in self.get_symbols_async(self.root.assets_list_symbols):
            time_start = process_time()
            candle_update = await self._fetch_candles(symbol, timeframe)
            if len(candle_update) > 0:
                market_history_update[symbol] = append_market_history_update(
                    market_history_update, candle_update, symbol
                )

            time_elapsed_difference = 1 - (process_time() - time_start)
            if time_elapsed_difference > 0:
                await asyncio_sleep(time_elapsed_difference)

        return market_history_update

    def convert_market_history_to_df(
        self, history_update, timeframe: None | dict[str, pd.Timestamp] = None
    ) -> None | pd.DataFrame:
        if not history_update:
            self.log.warning("No market update, cannot create dataframe!")
            return None
        history_df_list = []
        if timeframe is not None:
            history_df_list.append(get_reference_index(timeframe, freq=self.config.candle_interval))

        for asset_hist in history_update.values():
            if len(asset_hist) > 0:
                df_hist = pd.DataFrame(asset_hist)
                df_hist.sort_values("t", inplace=True, ascending=True)
                history_df_list.append(df_hist)

        df_market_history_update = reduce(
            lambda df_left, df_right: pd.merge_asof(df_left, df_right, on="t", direction="nearest", tolerance=60000),
            history_df_list,
        )
        df_market_history_update.set_index("t", inplace=True)
        return df_market_history_update

    async def update_market_history(self, start=None, end=None, init_timespan=None):
        if init_timespan is not None:
            latest_remote_candle_timestamp = await self.root.exchange_api.get_latest_remote_candle_timestamp()
            start = get_time_minus_delta(timestamp=latest_remote_candle_timestamp, delta=init_timespan)["timestamp"]

        if self.root.assets_list_symbols is None:
            self.root.assets_list_symbols = await self.root.exchange_api.get_active_assets()

        timeframe = await self.get_timeframe(start=start, end=end)

        if timeframe["ms_until_wait_over"] > 0:
            market_history_update = await self.fetch_market_history(timeframe["timeframe"])
            if len(market_history_update) > 1:
                df_market_history_update = self.convert_market_history_to_df(
                    market_history_update, timeframe=timeframe["timeframe"]
                )
            else:
                self.log.warning(
                    "No market update since %s. Maybe exchange down?",
                    str(pd.to_timedelta(timeframe["ms_until_wait_over"], unit="ms")),
                )
        else:
            self.log.info(
                "Update request too early. Frequency is %s. Wait at least %s min for next try!",
                self.config.candle_interval,
                int(abs(timeframe["ms_until_wait_over"]) / 1000 // 60),
            )
        return df_market_history_update
