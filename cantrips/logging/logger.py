import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from time import time
import numpy as np

import yaml
import os

from cantrips.configs import load_config
from cantrips.debugging.terminal import poem


class TruncateAndAlignFormatter(logging.Formatter):
    def __init__(self, *args, max_module_length=20, max_func_length=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_module_length = max_module_length
        self.max_func_length = max_func_length

    def format(self, record):
        module = poem(record.module, 20)
        lineno = f"ln{record.lineno:<3}"
        funcName = poem(record.funcName, 10)
        level = f"{record.levelname:<7}"
        message = record.getMessage()
        return f"{module} - {lineno} - {funcName} - {level}: {message}"


def get_logger(level="INFO"):
    np.set_printoptions(precision=2, suppress=True)
    # Suppress third-party logs
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).setLevel(logging.ERROR)
    
    config = load_config()

    basename = os.path.basename(sys.argv[0]).replace(".py", "")
    log_filename = f"{int(time()) - 1719520335}_{basename}_{os.getpid()}.log"

    logger = logging.getLogger(basename)

    # Check if handlers are already added
    if logger.hasHandlers():
        # Ensure logger level is set correctly if reusing
        logger.setLevel(eval(f"logging.{level}"))
        return logger

    handlers = [logging.StreamHandler()]
    if config.file_logging:
        handlers.append(logging.FileHandler(Path(config.filepath) / log_filename))

    # Define the truncation and alignment formatter
    formatter = TruncateAndAlignFormatter(
        max_module_length=20, 
        max_func_length=20
    )

    # Assign the formatter to each handler and add handlers to the logger
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(eval(f"logging.{level}"))

    return logger


@contextmanager
def suppress_all_output():
    """
    Context manager to suppress all stdout and stderr output.
    """
    # Save original file descriptors for stdout and stderr
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)
    # Open /dev/null
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        # Redirect stdout and stderr to /dev/null
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        # Restore original stdout and stderr
        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)
        os.close(devnull)
        os.close(original_stdout_fd)
        os.close(original_stderr_fd)

@contextmanager
def shht():
    logger = logging.getLogger()
    previous_level = logger.getEffectiveLevel()
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(previous_level)
