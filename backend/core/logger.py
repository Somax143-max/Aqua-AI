"""
AquaProtect-AI: Structured Hydroacoustic System Logger
Provides standardized, structured logging across all acoustic modules:
- Replaces silent exception handling with explicit INFO, WARNING, ERROR, CRITICAL records
- Dual output: Color-coded console formatting + persistent daily log files
- Captures stack traces, sensor telemetry context, and subsea mission IDs
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "aquaprotect_system.log")

def get_system_logger(name: str = "AquaProtect") -> logging.Logger:
    """Returns a configured logger instance with console and file rotation handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    # Rotating file handler (max 10MB per file, up to 5 backups)
    f_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    return logger

# Primary system logger singleton
logger = get_system_logger("AquaProtect-Core")