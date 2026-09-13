"""Ultron - logging setup."""
import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_loggers = {}

def get_logger(name: str) -> logging.Logger:
    """Get a logger with console + rotating file handlers."""
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(ch)
    
    # File handler
    fh = RotatingFileHandler(
        os.path.join(LOG_DIR, "ultron.log"),
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    
    # Error file handler
    eh = RotatingFileHandler(
        os.path.join(LOG_DIR, "ultron.err.log"),
        maxBytes=2_000_000,
        backupCount=2,
        encoding="utf-8",
    )
    eh.setLevel(logging.ERROR)
    eh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(eh)
    
    _loggers[name] = logger
    return logger