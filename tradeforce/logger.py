import os

import logging
from tqdm.auto import tqdm

# from tqdm import TqdmExperimentalWarning, TqdmSynchronisationWarning

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.buffered_records = []

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end="")
            self.flush()
        except (ValueError, TypeError):
            self.handleError(record)

    def flush_buffer(self):
        for record in self.buffered_records:
            self.emit(record)
        self.buffered_records.clear()

    def buffer(self, record):
        self.buffered_records.append(record)

    def handle(self, record):
        if record.levelno < self.level:
            return
        if tqdm.get_lock()._is_owned():
            self.buffer(record)
        else:
            self.flush_buffer()
            self.emit(record)


# class TqdmLoggingHandler(logging.StreamHandler):
#     def emit(self, record):
#         try:
#             msg = self.format(record)
#             tqdm.write(msg, end=self.terminator)
#         except (KeyboardInterrupt, SystemExit):
#             raise
#         except TqdmSynchronisationWarning:
#             pass
#         except TqdmExperimentalWarning:
#             pass
#         except (ValueError, TypeError):
#             self.handleError(record)


class Logger:
    def __init__(self):
        self.logging = logging.getLogger(__name__)
        self.logging.addHandler(TqdmLoggingHandler())
        # logger = logging.getLogger(__name__)
        # logging
        # logging.addHandler(TqdmLoggingHandler())

    def get_logger(self, name_space) -> logging.Logger:
        logger = logging.getLogger(name_space)

        return logger
