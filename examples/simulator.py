"""_summary_
"""


import numpy as np

import numba as nb  # type: ignore
from tradeforce import TradingEngine

tradeforce_config = {
    "trader": {
        "id": 1,
        "run_live": False,
        "budget": 10000,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
            "amount_invest_fiat": 100,
            "investment_cap": 0,
            "buy_opportunity_factor": 0.10,
            "buy_opportunity_boundary": 0.05,
            "prefer_performance": 1,
            "hold_time_limit": 1000,
            "profit_factor": 1.10,
            "profit_ratio_limit": 1.01,
            "moving_window_hours": 180,
        },
    },
    "backend": {
        "dbms": "mongodb",
        "dbms_host": "localhost",
        "dbms_port": 27017,
        "dbms_user": "tradeforce",
        "dbms_pw": "tradeforce",
        "dbms_connect_db": "admin",
        "local_cache": True,
    },
    # "backend": {
    #     "dbms": "postgresql",
    #     "dbms_host": "localhost",
    #     "dbms_port": 5432,
    #     "dbms_connect_db": "postgres",
    #     "dbms_user": "postgres",
    #     "dbms_pw": "postgres",
    #     "local_cache": False,
    # },
    "market_history": {
        "name": "test_simulator",
        "exchange": "bitfinex",
        "base_currency": "USD",
        "candle_interval": "5min",
        "history_timeframe": "60days",
        "update_mode": "none",
        # "force_source": "local_cache",
    },
    "simulation": {
        "snapshot_size": 100000,
        "snapshot_amount": 10,
    },
}


def pre_process(config, market_history):
    sim_start_delta = config.sim_start_delta
    asset_prices = market_history.get_market_history(start=sim_start_delta, metrics=["o"], fill_na=True)

    asset_prices_pct = market_history.get_market_history(
        start=sim_start_delta, metrics=["o"], fill_na=True, pct_change=True, pct_as_factor=False, pct_first_row=0
    )

    lower_threshold = 0.000005
    upper_threshold = 0.999995
    quantiles = asset_prices_pct.stack().quantile([lower_threshold, upper_threshold])
    asset_prices_pct[asset_prices_pct > quantiles[upper_threshold]] = quantiles[upper_threshold]
    asset_prices_pct[asset_prices_pct < quantiles[lower_threshold]] = quantiles[lower_threshold]

    moving_window_increments = int(config.moving_window_increments)
    asset_market_performance = asset_prices_pct.rolling(
        window=moving_window_increments, step=1, min_periods=moving_window_increments
    ).sum(engine="numba", engine_kwargs={"parallel": True, "cache": True})[moving_window_increments - 1 :]

    preprocess_return = {
        "asset_prices": asset_prices,
        "asset_prices_pct": asset_prices_pct,
        "asset_performance": asset_market_performance,
    }
    return preprocess_return


@nb.njit(cache=True, parallel=False)
def buy_strategy(params, df_asset_prices_pct, df_asset_performance):
    row_idx = np.int64(params["row_idx"] - params["moving_window_increments"])  # init row_idx == 0
    buyfactor_row = df_asset_performance[row_idx]
    # moving_window_increments_history_prices_pct = get_current_moving_window_increments(params, df_asset_prices_pct)
    # buyfactor_row = np.sum(moving_window_increments_history_prices_pct, axis=0)

    buy_opportunity_factor_min = params["buy_opportunity_factor"] - params["buy_opportunity_boundary"]
    buy_opportunity_factor_max = params["buy_opportunity_factor"] + params["buy_opportunity_boundary"]
    buy_options_bool = (buyfactor_row >= buy_opportunity_factor_min) & (buyfactor_row <= buy_opportunity_factor_max)
    if np.any(buy_options_bool):
        buy_option_indices = np.where(buy_options_bool)[0].astype(np.float64)
        buy_option_values = buyfactor_row[buy_options_bool]
        # prefer_performance can be -1, 1, and 0.
        if params["prefer_performance"] == 0:
            buy_option_values = np.absolute(buy_option_values - params["buy_opportunity_factor"])
        buy_option_array = np.vstack((buy_option_indices, buy_option_values))
        buy_option_array = buy_option_array[:, buy_option_array[1, :].argsort()]
        if params["prefer_performance"] == 1:
            # flip axis
            buy_option_array = buy_option_array[:, ::-1]
        buy_option_array_int = buy_option_array[0].astype(np.int64)
    return buy_option_array_int


@nb.njit(cache=True, parallel=False)
def sell_strategy(params, buybag, history_prices_row):
    buy_option_idxs = buybag[:, 0:1].T.flatten().astype(np.int64)
    prices_current = history_prices_row[buy_option_idxs].reshape((1, -1)).T
    prices_profit = buybag[:, 4:5]

    # check plausibility and prevent false logic
    # profit gets a max plausible threshold
    sanity_check_mask = (prices_current / prices_profit > 1.2).flatten()
    prices_current[sanity_check_mask] = prices_profit[sanity_check_mask]
    times_since_buy = params["row_idx"] - buybag[:, 2:3]
    current_profit_ratios = prices_current / buybag[:, 3:4]
    sell_prices_reached = prices_current >= prices_profit
    ok_to_sells = (times_since_buy > params["hold_time_limit"]) & (
        current_profit_ratios >= params["profit_ratio_limit"]
    )
    sell_assets = (sell_prices_reached | ok_to_sells).flatten()
    return sell_assets, prices_current


sim_result = TradingEngine(config=tradeforce_config).run_sim(pre_process=None, buy_strategy=None, sell_strategy=None)
print(sim_result["profit"])
