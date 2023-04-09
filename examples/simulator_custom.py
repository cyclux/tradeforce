"""_summary_
"""


import numpy as np
import numba as nb  # type: ignore

from tradeforce import Tradeforce

CONFIG = {
    "trader": {
        "budget": 10000,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
            "amount_invest_per_asset": 100,
            "investment_cap": 0,
            "buy_performance_score": 0.10,
            "buy_performance_boundary": 0.05,
            "buy_performance_preference": 1,
            "_hold_time_increments": 1000,
            "profit_factor_target": 1.10,
            "profit_factor_target_min": 1.01,
            "moving_window_hours": 180,
        },
    },
    "backend": {
        "dbms": "postgresql",
        "dbms_host": "docker_postgres",
        "dbms_port": 5432,
        "dbms_connect_db": "postgres",
        "dbms_user": "postgres",
        "dbms_pw": "postgres",
        "local_cache": False,
    },
    "market_history": {
        "name": "bfx_history_docker_test",
        "exchange": "bitfinex",
        "base_currency": "USD",
        "candle_interval": "5min",
        "history_timeframe_days": 60,
        "update_mode": "none",
        # "force_source": "local_cache",
    },
    "simulation": {
        "_subset_size_increments": 10000,
        "subset_amount": 10,
    },
}


def pre_process(config, market_engine):
    asset_prices = market_engine.get_market_history(metrics=["o"], fill_na=True)

    asset_prices_pct = market_engine.get_market_history(
        metrics=["o"],
        fill_na=True,
        pct_change=True,
        pct_as_factor=False,
        pct_first_row=0,
    )

    lower_threshold = 0.000005
    upper_threshold = 0.999995
    quantiles = asset_prices_pct.stack().quantile([lower_threshold, upper_threshold])
    asset_prices_pct[asset_prices_pct > quantiles[upper_threshold]] = quantiles[upper_threshold]
    asset_prices_pct[asset_prices_pct < quantiles[lower_threshold]] = quantiles[lower_threshold]

    _moving_window_increments = int(config._moving_window_increments)
    asset_market_performance = asset_prices_pct.rolling(
        window=_moving_window_increments, step=1, min_periods=_moving_window_increments
    ).sum(engine="numba", engine_kwargs={"parallel": True, "cache": True})[_moving_window_increments - 1 :]

    preprocess_return = {
        "asset_prices": asset_prices,
        "asset_prices_pct": asset_prices_pct,
        "asset_performance": asset_market_performance,
    }
    return preprocess_return


@nb.njit(cache=True, parallel=False)
def buy_strategy(params, df_asset_prices_pct, df_asset_performance):
    row_idx = np.int64(params["row_idx"] - params["_moving_window_increments"])  # init row_idx == 0
    buyfactor_row = df_asset_performance[row_idx]
    # _moving_window_increments_history_prices_pct = get_current__moving_window_increments(params, df_asset_prices_pct)
    # buyfactor_row = np.sum(_moving_window_increments_history_prices_pct, axis=0)

    buy_performance_score_min = params["buy_performance_score"] - params["buy_performance_boundary"]
    buy_performance_score_max = params["buy_performance_score"] + params["buy_performance_boundary"]
    buy_options_bool = (buyfactor_row >= buy_performance_score_min) & (buyfactor_row <= buy_performance_score_max)
    if np.any(buy_options_bool):
        buy_option_indices = np.where(buy_options_bool)[0].astype(np.float64)
        buy_option_values = buyfactor_row[buy_options_bool]
        # buy_performance_preference can be -1, 1, and 0.
        if params["buy_performance_preference"] == 0:
            buy_option_values = np.absolute(buy_option_values - params["buy_performance_score"])
        buy_option_array = np.vstack((buy_option_indices, buy_option_values))
        buy_option_array = buy_option_array[:, buy_option_array[1, :].argsort()]
        if params["buy_performance_preference"] == 1:
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
    ok_to_sells = (times_since_buy > params["_hold_time_increments"]) & (
        current_profit_ratios >= params["profit_factor_target_min"]
    )
    sell_assets = (sell_prices_reached | ok_to_sells).flatten()
    return sell_assets, prices_current


def main():
    sim_result = Tradeforce(config=CONFIG).run_sim(
        pre_process=pre_process, buy_strategy=buy_strategy, sell_strategy=sell_strategy
    )
    print(sim_result["profit"])


if __name__ == "__main__":
    main()
