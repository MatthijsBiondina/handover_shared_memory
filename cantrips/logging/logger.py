import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from time import time

import yaml

import logging
import os

from cantrips.configs import load_config


def get_logger():
    config = load_config()

    basename = os.path.basename(sys.argv[0]).replace(".py", "")
    log_filename = f"{int(time()) - 1719520335}_{basename}_{os.getpid()}.log"

    logger = logging.getLogger(basename)

    handlers = [logging.StreamHandler()]
    if config.file_logging:
        handlers.append(logging.FileHandler(Path(config.filepath) / log_filename))

    # Configure the basic logging settings
    logging.basicConfig(
        level=eval(f"logging.{config.level}"),
        format=config.format,
        handlers=handlers,
    )

    logger.setLevel(eval(f"logging.{config.level}"))

    return logger

@contextmanager
def shht():
    logger = logging.getLogger()
    previous_level=logger.getEffectiveLevel()
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(previous_level)
