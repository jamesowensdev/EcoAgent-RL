import logging
import os
import sys
from datetime import datetime

def setup_logger(name: str, log_level=logging.INFO, log_to_file=True):
    logger = logging.getLogger(name)

    if log_to_file:
        os.makedirs("logs", exist_ok=True)

    if logger.hasHandlers():
        return logger

    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        log_dir = "logs"
        file_name = f"{log_dir}/sim_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
