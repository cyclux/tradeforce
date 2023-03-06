"""_summary_

Returns:
    _type_: _description_
"""


import numpy as np
import pandas as pd

from tradeforce.utils import get_df_datetime_index

##################
# DB interaction #
##################


def drop_dict_na_values(record):
    return {key: record[key] for key in record if not pd.isna(record[key])}


class Backend:
    """Fetch market history from local or remote database and store in DataFrame"""

    def __init__(self, root):
        self.root = root
        self.config = root.config
        self.log = root.logging.getLogger(__name__)
        self.sync_check_needed = False
        self.is_collection_new = True
        self.is_filled_na = False

    def get_internal_db_index(self):
        return self.root.market_history.df_market_history.sort_index().index

    def check_db_consistency(self):
        is_consistent = False
        index = self.get_internal_db_index()
        timeframe = {
            "start_datetime": pd.Timestamp(index[0], unit="ms", tz="UTC"),
            "end_datetime": pd.Timestamp(index[-1], unit="ms", tz="UTC"),
        }
        real_index = get_df_datetime_index(timeframe, freq=self.config.candle_interval)["t"].to_list()
        current_index = index.to_list()
        index_diff = np.setdiff1d(real_index, current_index)
        if len(index_diff) > 0:
            self.log.warning("Inconsistent asset history. Missing candle timestamps: %s", str(index_diff))
            # TODO: fetch missing candle timestamps (index_diff) from remote
        else:
            is_consistent = True
            self.log.info("Consistency check of DB history successful!")
        return is_consistent

    #############
    # DB checks #
    #############

    def db_sync_check(self):
        internal_db_index = np.array(self.get_internal_db_index())

        external_db_index = self.mongo_exchange_coll.find({}, projection={"_id": False, "t": True}).sort("t", 1)
        external_db_index = np.array([index_dict["t"] for index_dict in list(external_db_index)])

        only_exist_in_external_db = np.setdiff1d(external_db_index, internal_db_index)
        only_exist_in_internal_db = np.setdiff1d(internal_db_index, external_db_index)

        sync_from_external_db_needed = len(only_exist_in_external_db) > 0
        sync_from_internal_db_needed = len(only_exist_in_internal_db) > 0

        if sync_from_external_db_needed:
            external_db_entries_to_sync = self.mongo_exchange_coll.find(
                {"t": {"$in": only_exist_in_external_db.tolist()}}, projection={"_id": False}
            )
            df_external_db_entries = pd.DataFrame(list(external_db_entries_to_sync)).set_index("t")
            self.root.market_history.df_market_history = pd.concat(
                [self.root.market_history.df_market_history, df_external_db_entries]
            ).sort_values("t")
            self.log.info(
                "%s candles synced from external to internal DB (%s to %s)",
                len(only_exist_in_external_db),
                min(only_exist_in_external_db),
                max(only_exist_in_external_db),
            )

        if sync_from_internal_db_needed:
            internal_db_entries = self.root.market_history.get_market_history(from_list=only_exist_in_internal_db)
            internal_db_entries_to_sync = [
                drop_dict_na_values(record) for record in internal_db_entries.reset_index(drop=False).to_dict("records")
            ]
            self.mongodb_insert(internal_db_entries_to_sync)
            self.log.info(
                "%s candles synced from internal to external DB (%s to %s)",
                len(only_exist_in_internal_db),
                min(only_exist_in_internal_db),
                max(only_exist_in_internal_db),
            )

        self.log.info("Internal and external DB are synced")
        if self.config.force_source == "local_cache" and sync_from_external_db_needed:
            self.root.market_history.local_cache()
