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

# ANSI escape codes
BOLD = '\033[1m'
RESET = '\033[0m'

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
        # Add bold formatting to the entire log message
        return f"{BOLD}{module} - {lineno} - {funcName} - {level}: {message}{RESET}"


class BoldStreamHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()
        # Check if the output stream supports ANSI codes
        self.supports_color = hasattr(self.stream, 'isatty') and self.stream.isatty()

    def format(self, record):
        # Only add ANSI codes if the stream supports them
        if self.supports_color:
            return super().format(record)
        else:
            # Strip ANSI codes if not supported
            msg = super().format(record)
            return msg.replace(BOLD, '').replace(RESET, '')


def get_logger(level="INFO"):
    np.set_printoptions(precision=2, suppress=True)
    # Suppress third-party logs
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).setLevel(logging.ERROR)
    
    config = load_config()

    basename = os.path.basename(sys.argv[0]).replace(".py", "")
    log_filename = f"{int(time()) - 1719520335}_{basename}_{os.getpid()}.log"

    logger = logging.getLogger(basename)

    if logger.hasHandlers():
        logger.setLevel(eval(f"logging.{level}"))
        return logger

    # Use BoldStreamHandler instead of regular StreamHandler
    handlers = [BoldStreamHandler()]
    if config.file_logging:
        # File handler doesn't need ANSI codes
        file_handler = logging.FileHandler(Path(config.filepath) / log_filename)
        handlers.append(file_handler)

    formatter = TruncateAndAlignFormatter(
        max_module_length=20, 
        max_func_length=20
    )

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(eval(f"logging.{level}"))

    return logger