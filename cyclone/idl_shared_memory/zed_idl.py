from dataclasses import dataclass
import numpy as np

from cyclone.defaults import Config
from cyclone.idl_shared_memory.base_idl import BaseIDL


@dataclass
class ZEDIDL(BaseIDL):
    timestamp: np.ndarray = np.empty((1,), dtype=np.float64)
    color: np.ndarray = np.empty((Config.H_zed, Config.W_zed, 3), dtype=np.uint8)
    depth: np.ndarray = np.empty((Config.H_zed, Config.W_zed), dtype=np.float32)
    extrinsics: np.ndarray = np.empty((4, 4), dtype=np.float64)
    intrinsics: np.ndarray = np.empty((3, 3), dtype=np.float64)