from dataclasses import dataclass

import numpy as np

from cyclone.defaults import Config
from cyclone.idl_shared_memory.base_idl import BaseIDL


@dataclass
class YOLOIDL(BaseIDL):
    timestamp: np.ndarray = np.empty((1,), dtype=np.float64)
    # color: np.ndarray = np.empty((Config.H_d405, Config.W_d405, 3), dtype=np.uint8)
    # depth: np.ndarray = np.empty((Config.H_d405, Config.W_d405), dtype=np.uint16)
    # points: np.ndarray = np.empty((Config.H_d405, Config.W_d405, 3), dtype=np.float32)
    extrinsics: np.ndarray = np.empty((4, 4), dtype=np.float64)
    intrinsics: np.ndarray = np.empty((3, 3), dtype=np.float64)

    objects: np.ndarray = np.empty((16, 6), dtype=np.float32)
    uv: np.ndarray=np.empty((16, 2), dtype=int)
    xyz: np.ndarray = np.empty((16, 3), dtype=np.float32)