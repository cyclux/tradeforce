import sys

# import pandas as pd
import psycopg2
from urllib.parse import quote_plus, urlparse
from tradeforce.backend import Backend


class BackendSQL(Backend):
    def __init__(self, root):
        super().__init__(root)
        self.dbms_client = self.connect_dbms_mongo()
        # Only sync backend now if there is no exchange API connection.
        # In case an API connection is used, db_sync_trader_state()
        # will be called once by exchange_ws -> ws_priv_wallet_snapshot()
        if self.config.use_dbms and not self.config.run_live:
            self.db_sync_trader_state()

    def construct_uri_sql(self):
        if self.config.dbms_user and self.config.dbms_pw:
            dbms_uri = (
                f"postgresql://{quote_plus(self.config.dbms_user)}"
                + f":{quote_plus(self.config.dbms_pw)}"
                + f"@{self.config.dbms_host}"
                + f":{self.config.dbms_port}"
            )
        else:
            dbms_uri = f"postgresql://{self.config.dbms_host}:{self.config.dbms_port}"
        return dbms_uri

    def connect_backend_sql(self):
        dbms_uri = self.construct_uri_sql()
        conncection_options = urlparse(dbms_uri)
        print(conncection_options)
        dbms_client = psycopg2.connect(**conncection_options)
        self.dbms_db = dbms_client.cursor()
        self.dbms_db.execute(f"CREATE DATABASE IF NOT EXISTS {self.config.dbms_database}")
        self.dbms_db.execute(f"USE {self.config.dbms_database}")
        self.dbms_db.execute(
            f"CREATE TABLE IF NOT EXISTS {self.config.dbms_table_or_coll_name} (id SERIAL PRIMARY KEY, data JSONB)"
        )
        self.dbms_db.connection.commit()
        try:
            self.dbms_db.execute("SELECT 1")
        except psycopg2.OperationalError as exc:
            self.log.error(
                "Failed connecting to PostgreSQL (%s). Is the PostgreSQL instance running and reachable?",
                self.config.dbms_host,
            )
            self.log.exception(exc)
        else:
            self.log.info("Successfully connected to PostgreSQL backend (%s)", self.config.dbms_host)

            try:
                self.dbms_db.execute(f"SELECT COUNT(*) FROM {self.config.dbms_table_or_coll_name}")
            except psycopg2.ProgrammingError:
                self.is_collection_new = True
            else:
                self.is_collection_new = False
                if self.config.force_source != "postgresql":
                    if self.config.force_source == "api":
                        sys.exit(
                            f"[ERROR] PostgreSQL history '{self.config.dbms_table_or_coll_name}' already exists. "
                            + "Cannot load history via API. Choose different history DB name or loading method."
                        )
                    self.sync_check_needed = True
        return dbms_client
