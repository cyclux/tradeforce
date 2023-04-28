""" logger.py

Module: tradeforce.logger
-------------------------

Provides a custom logging handler, TqdmLoggingHandler.

It integrates logging with tqdm progress bars, ensuring that log messages
and progress bar updates do not interfere with each other.
The module depends on the standard logging library and the tqdm library.

The TqdmLoggingHandler class buffers log records when a tqdm progress bar is active
and flushes them when the progress bar is not locked. The handler inherits from the
logging.Handler class and provides methods to emit, buffer, and handle log records,
as well as to flush the buffered log records.

The Logger class is a wrapper to create and configure loggers with the TqdmLoggingHandler.
It provides a method to get a logger with a specified namespace,
creating and returning a logger with the given namespace and the custom handler attached.

Key features:
    Custom logging handler to integrate logging with tqdm progress bars
    Buffering of log records when a progress bar is active
    Flushing buffered log records when the progress bar is not locked
    Logger wrapper class to create and configure loggers with the custom handler

Dependencies:
    Python standard logging library
    tqdm library

"""

import os
import logging
from logging import LogRecord

from tqdm.auto import tqdm

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler to integrate logging with tqdm progress bars.

    Inherits from the Python logging.Handler class.
    It buffers log records when a tqdm progress bar is active and flushes them when the progress bar is not locked.
    """

    def __init__(self, level: int = logging.NOTSET) -> None:
        """Initialize the TqdmLoggingHandler instance.

        Params:
            level: The minimum logging level (e.g., logging.INFO) at which this handler processes log records.
        """
        super().__init__(level)
        self.buffered_records: list = []

    def emit(self, record: LogRecord) -> None:
        """Emit a log record.

        Formats the log record and writes it to the tqdm progress bar.
        Handles exceptions that may occur during formatting and writing.

        Params:
            record: The log record to be emitted.
        """
        try:
            msg = self.format(record)
            tqdm.write(msg, end="")
            self.flush()
        except (ValueError, TypeError):
            self.handleError(record)

    def flush_buffer(self) -> None:
        """Flush the buffered log records.

        Emits all buffered records and clears the buffer.
        """
        for record in self.buffered_records:
            self.emit(record)
        self.buffered_records.clear()

    def buffer(self, record: LogRecord) -> None:
        """Buffer a log record.

        Append a log record to the buffer.

        Params:
            record: The log record to be buffered.
        """
        self.buffered_records.append(record)

    def handle(self, record: LogRecord) -> bool:
        """Handle a log record.

        If the record's logging level is below the handler's level, it will not be processed.
        If a tqdm progress bar is active (locked), buffer the record.
        Otherwise, flush the buffer and emit the record.

        Params:
            record: The log record to be handled.
        """
        if record.levelno < self.level:
            return False
        if tqdm.get_lock()._is_owned():
            self.buffer(record)
        else:
            self.flush_buffer()
            self.emit(record)
        return True


class Logger:
    """A wrapper class to create and configure loggers with TqdmLoggingHandler."""

    def __init__(self) -> None:
        """Initialize the Logger instance.

        Creates and configures the root logger with TqdmLoggingHandler.
        """
        self.logging = logging.getLogger(__name__)
        self.logging.addHandler(TqdmLoggingHandler())

    def get_logger(self, name_space: str) -> logging.Logger:
        """Get a logger with a specified namespace.

        Creates and returns a logger for the given namespace.

        Params:
            name_space: The namespace for the logger.

        Returns:
            A logging.Logger instance with the given namespace.
        """
        logger = logging.getLogger(name_space)

        return logger
