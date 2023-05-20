Configuration options
=====================

.. automodule:: tradeforce.config

.. autoclass:: tradeforce.config.Config

   ----

   core
   ----

   .. autoattribute:: tradeforce.config.Config.working_dir
   .. autoattribute:: tradeforce.config.Config.credentials_path

   --------------

   market_history
   --------------

   Determines how the market data is stored, updated, and retrieved. This includes specifying
   the entity and file names for data storage and defining the base currency.
   It also includes options for the frequency and mode of data updates, checks for database
   synchronization and consistency, and the choice of preferred data source on start.

   .. autoattribute:: tradeforce.config.Config.name
   .. autoattribute:: tradeforce.config.Config.exchange
   .. autoattribute:: tradeforce.config.Config.base_currency
   .. autoattribute:: tradeforce.config.Config.candle_interval
   .. autoattribute:: tradeforce.config.Config.update_mode
   .. autoattribute:: tradeforce.config.Config.check_db_sync
   .. autoattribute:: tradeforce.config.Config.check_db_consistency
   .. autoattribute:: tradeforce.config.Config.force_source

   -----------------------

   initial_relevant_assets
   -----------------------

   Determines the parameters and constraints for initial loading and filtering of
   relevant assets for simulation and trading. It includes settings that define
   the timeframe for historical data fetching, the maximum number of assets to load,
   and criteria for asset relevancy based on candle data attributes (amount and sparsity).
   There is also an option to exclude specific assets, such as stable coins or those
   deemed unsuitable for trading.

   .. autoattribute:: tradeforce.config.Config.fetch_init_timeframe_days
   .. autoattribute:: tradeforce.config.Config.relevant_assets_cap
   .. autoattribute:: tradeforce.config.Config.relevant_assets_max_candle_sparsity
   .. autoattribute:: tradeforce.config.Config.relevant_assets_min_amount_candles
   .. autoattribute:: tradeforce.config.Config.assets_excluded

   -------

   backend
   -------

   Determines the Database Management System (DBMS) used for storing market history
   and trader-specific data. It includes options for selecting the DBMS, with support
   for Postgres and MongoDB, and defining the host and port for the DBMS connection.
   The specified database name will be created if it doesn't exist. Additionally,
   provides the option to save the in-memory database to a local `.arrow` file for faster
   loading in future runs (especially useful in hyperparam optimizations with many
   parallel simulations).

   .. autoattribute:: tradeforce.config.Config.dbms
   .. autoattribute:: tradeforce.config.Config.dbms_host
   .. autoattribute:: tradeforce.config.Config.dbms_port
   .. autoattribute:: tradeforce.config.Config.dbms_db
   .. autoattribute:: tradeforce.config.Config.dbms_connect_db
   .. autoattribute:: tradeforce.config.Config.dbms_user
   .. autoattribute:: tradeforce.config.Config.dbms_pw
   .. autoattribute:: tradeforce.config.Config.local_cache

   ------

   trader
   ------

   Determines the trader parameters which include the live trading mode, setting of unique identifiers
   for trader instances, and defining the initial trading budget for simulations.
   Settings apply to both, live trading and simulations.

   .. autoattribute:: tradeforce.config.Config.run_live
   .. autoattribute:: tradeforce.config.Config.trader_id
   .. autoattribute:: tradeforce.config.Config.budget
   .. autoattribute:: tradeforce.config.Config.maker_fee
   .. autoattribute:: tradeforce.config.Config.taker_fee

   --------

   strategy
   --------

   Determines the parameters of the default trading strategies (See :doc:`default_strategies`)
   which is applied to both, live trading and simulations. The optimal values of those parameters
   can be found by defining a search space and running simulations via Optuna hyperparameter
   tuning. For more information and an example see `hyperparam_search.py`_.

   .. autoattribute:: tradeforce.config.Config.amount_invest_per_asset
   .. autoattribute:: tradeforce.config.Config.moving_window_hours
   .. autoattribute:: tradeforce.config.Config.buy_signal_score
   .. autoattribute:: tradeforce.config.Config.buy_signal_boundary
   .. autoattribute:: tradeforce.config.Config.buy_signal_preference
   .. autoattribute:: tradeforce.config.Config.profit_factor_target
   .. autoattribute:: tradeforce.config.Config.hold_time_days
   .. autoattribute:: tradeforce.config.Config.profit_factor_target_min
   .. autoattribute:: tradeforce.config.Config.investment_cap
   .. autoattribute:: tradeforce.config.Config.max_buy_per_asset

   ----------

   simulation
   ----------

   Determines simulation specific parameters.

   .. autoattribute:: tradeforce.config.Config.subset_size_days
   .. autoattribute:: tradeforce.config.Config.subset_amount
   .. autoattribute:: tradeforce.config.Config.train_val_split_ratio

   -------

   logging
   -------

   Sets the logging level for the different loggers.

   .. autoattribute:: tradeforce.config.Config.log_level_ws_live
   .. autoattribute:: tradeforce.config.Config.log_level_ws_update

.. _hyperparam_search.py: https://github.com/cyclux/tradeforce/blob/master/examples/hyperparam_search.py