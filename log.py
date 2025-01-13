"""Logging configuration for """

from __future__ import annotations

from contextlib import contextmanager
from io import StringIO
import logging
from queue import Queue
import threading
import time
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    from slashed import OutputWriter


class LogCapturer:
    """Captures log output for display in UI."""

    def __init__(self):
        """Initialize log capturer."""
        self.log_queue: Queue[str] = Queue()
        self.buffer = StringIO()
        self.handler = logging.StreamHandler(self.buffer)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.handler.setFormatter(fmt)

    def start(self):
        """Start capturing logs."""
        logging.getLogger().addHandler(self.handler)

        def monitor():
            while True:
                if self.buffer.tell():
                    self.buffer.seek(0)
                    self.log_queue.put(self.buffer.getvalue())
                    self.buffer.truncate(0)
                    self.buffer.seek(0)
                time.sleep(0.1)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop capturing logs."""
        logging.getLogger().removeHandler(self.handler)

    def get_logs(self) -> str:
        """Get accumulated logs."""
        logs = []
        while not self.log_queue.empty():
            logs.append(self.log_queue.get_nowait())
        return "".join(logs)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given name.

    Args:
        name: The name of the logger, will be prefixed with ''

    Returns:
        A logger instance
    """
    return logging.getLogger(f"{name}")


@contextmanager
def set_handler_level(
    level: int,
    logger_names: Sequence[str],
    *,
    session_handler: OutputWriter | None = None,
):
    """Temporarily set logging level and optionally add session handler.

    Args:
        level: Logging level to set
        logger_names: Names of loggers to configure
        session_handler: Optional output writer for session logging
    """
    loggers = [logging.getLogger(name) for name in logger_names]
    old_levels = [logger.level for logger in loggers]

    handler = None
    if session_handler:
        from slashed.log import SessionLogHandler

        handler = SessionLogHandler(session_handler)
        for logger in loggers:
            logger.addHandler(handler)

    try:
        for logger in loggers:
            logger.setLevel(level)
        yield
    finally:
        for logger, old_level in zip(loggers, old_levels):
            logger.setLevel(old_level)
            if handler:
                logger.removeHandler(handler)
