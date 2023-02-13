"""_summary_
"""
from fts import FastTradingSimulator
from fts_simulator import run_simulation

history_config = {
    "asset_interval": "5min",
    "history_timeframe": "750days",
    "base_currency": "USD",
    "exchange": "bitfinex",
    "load_history_via": "feather",
    "backend": "feather",
    "backend_host": "localhost:1234",
    "mongo_collection": "bfx_history_2y",
    "update_history": False,
    "run_exchange_api": False,
    "keep_updated": False,
}


fts = FastTradingSimulator(history_config).run()
bfx_history = fts.market_history.get_market_history(metrics=["o"], fill_na=True)
bfx_history_pct = fts.market_history.get_market_history(
    metrics=["o"], fill_na=True, pct_change=True, pct_as_factor=False
)

sim_params = {
    "window": 180,  # 190
    "buy_opportunity_factor": 0.10,
    "buy_opportunity_boundary": 0.05,
    "profit_factor": 1.7,  # 1.69
    "amount_invest_fiat": 100,
    "hold_time_limit": 1000,  # 4000
    "profit_ratio_limit": 1,  # 1
    "max_buy_per_asset": 1,
    "prefer_performance": "positive",
    "index_start": 0,
    "budget": 1100,
    "buy_limit": False,
    "exchange_fee": 0.15,
    "snapshot_size": -1,
    "snapshot_amount": 1,
}
# sim_total_profit, sim_trades_history, sim_buy_log
sim_total_profit, _, _ = run_simulation(
    sim_params,
    bfx_history,
    bfx_history_pct,
)
print(sim_total_profit)
