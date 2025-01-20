from dataclasses import dataclass
import numpy as np

from cyclone.defaults import Config
from cyclone.idl_shared_memory.base_idl import BaseIDL


@dataclass
class MasksIDL(BaseIDL):
    timestamp: np.ndarray = np.empty((1,), dtype=np.float64)
    color: np.ndarray = np.empty((Config.height, Config.width, 3), dtype=np.uint8)
    depth: np.ndarray = np.empty((Config.height, Config.width), dtype=np.uint16)

    mask_hand: np.ndarray = np.empty((Config.height, Config.width), dtype=np.bool_)
    mask_object: np.ndarray = np.empty((Config.height, Config.width), dtype=np.bool_)

    extrinsics: np.ndarray = np.empty((4, 4), dtype=np.float64)
    intrinsics: np.ndarray = np.empty((3, 3), dtype=np.float64)
    
