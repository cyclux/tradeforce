##############
Core Setup
##############
self.working_dir
self.check_db_consistency

##################
Logging Levels
##################
self.log_level_core
self.log_level_ws_live
self.log_level_ws_update

#######################
Database Connection
#######################
self.dbms
self.dbms_host
self.dbms_port
self.dbms_user
self.dbms_pw
self.dbms_connect_db
self.dbms_db
self.dbms_history_entity_name
self.local_cache

######################
Market Data & Updates
######################
self.exchange
self.base_currency
self.candle_interval
self.history_timeframe_days
self.force_source
self.update_mode

#####################
Exchange Settings
#####################
self.run_live
self.credentials_path
self.maker_fee
self.taker_fee

##########################
Strategy & Trader Settings
##########################
self.trader_id
self.moving_window_hours
self._moving_window_increments
self.budget
self.relevant_assets_cap
self.buy_performance_score
self.buy_performance_boundary
self.buy_performance_preference
self.profit_factor_target
self.profit_factor_target_min
self.hold_time_days
self._hold_time_increments
self.amount_invest_per_asset
self.max_buy_per_asset
self.investment_cap

###################
Simulator Config
###################
self.subset_size_days
self._subset_size_increments
self.subset_amount