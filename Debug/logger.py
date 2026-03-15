"""
Logger Module
=============
Configurable simulation logger wrapping Python's logging module.
"""

import logging
import sys
from typing import Optional


class SimLogger:
    """
    Simulation logger providing per-module logging with configurable levels.

    Usage
    -----
    >>> logger = SimLogger("Engine", level="DEBUG")
    >>> logger.info("Simulation started")
    >>> logger.debug("Tick 42: agents moved")
    """

    _loggers: dict = {}

    def __init__(
        self,
        name: str = "MAS_RMFS",
        level: str = "INFO",
        log_file: Optional[str] = None,
    ):
        # Reuse existing logger if already created for this name
        if name in SimLogger._loggers:
            self._logger = SimLogger._loggers[name]
            return

        self._logger = logging.getLogger(f"MAS_RMFS.{name}")
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.propagate = False

        # Console handler
        if not self._logger.handlers:
            fmt = logging.Formatter(
                fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(fmt)
            self._logger.addHandler(console_handler)

            # Optional file handler
            if log_file:
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setFormatter(fmt)
                self._logger.addHandler(file_handler)

        SimLogger._loggers[name] = self._logger

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)
