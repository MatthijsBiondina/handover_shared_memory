import inspect
from pathlib import Path
from typing import Dict, Any

import yaml
from munch import Munch


def load_config() -> Munch:
    caller_file = inspect.stack()[1].filename
    config_path: Path = Path(caller_file).parent / "config.yaml"
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return Munch.fromDict(config)
