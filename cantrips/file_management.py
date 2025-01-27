import os
from pathlib import Path
from typing import Union

from cantrips.debugging.terminal import pyout


def pyopen(path, mode):
    pyout(f"{mode} >> {os.path.abspath(path)}", color="BLUE")
    return open(path, mode)

def makedirs(path: Union[str, Path]):
    if isinstance(path, Path):
        parts = path.parts
    else:
        parts = path.split("/")

    if parts[0] == '':
        pth = Path("/")
    else:
        pth = Path(parts[0])
        os.makedirs(pth, exist_ok=True)
    for folder in parts[1:]:
        pth /= folder
        os.makedirs(pth, exist_ok=True)
    # pyout(f"mk >> {os.path.abspath(path)}", color="BLUE")

def listdir(path: Union[str, Path]):
    filenames = sorted(os.listdir(path))
    filepaths = [f"{path}/{fname}" for fname in filenames]
    filepaths = [Path(os.path.abspath(path)) for path in filepaths]
    return filepaths
