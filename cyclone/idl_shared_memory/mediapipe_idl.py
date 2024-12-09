from dataclasses import dataclass
import numpy as np
from cyclone.idl_shared_memory.base_idl import BaseIDL


@dataclass
class MediapipeIDL(BaseIDL):
    timestamp: np.ndarray = np.empty((1,), dtype=np.float64)
    color: np.ndarray = np.empty((480, 848, 3), dtype=np.uint8)
    depth: np.ndarray = np.empty((480, 848), dtype=np.uint16)
    points: np.ndarray = np.empty((480, 848, 3), dtype=np.float32)
    extrinsics: np.ndarray = np.empty((4, 4), dtype=np.float64)
    intrinsics: np.ndarray = np.empty((3, 3), dtype=np.float64)

    uv: np.ndarray = np.empty((8, 2), dtype=np.float32)
    xyz: np.ndarray = np.empty((8, 3), dtype=np.float32)

    # mask: np.ndarray = np.empty((480, 848), dtype=np.bool_)
    # centroid_uv: np.ndarray = np.empty((2,), dtype=np.int32)
    # centroid_xyz: np.ndarray = np.empty((3,), dtype=np.float32)
    # right_handedness: np.ndarray = np.empty((1,), dtype=np.bool_)
    # landmarks: np.ndarray = np.empty((33, 2), dtype=np.float32)
