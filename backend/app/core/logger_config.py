import os
import logging
from logging.handlers import RotatingFileHandler

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("backend")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )

    console_formatter = logging.Formatter(
        "%(levelname)s | %(message)s"
    )


    # ======================
    # CONSOLE
    # ======================

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)


    # ======================
    # FILE
    # ======================

    file_handler = RotatingFileHandler(
        "logs/backend.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)


    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


logger.propagate = False