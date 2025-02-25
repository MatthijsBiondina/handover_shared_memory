"""
This module defines the FrameBuffer class, which extends BufferTemplate to define
the structure of the data to be shared via shared memory.

Classes:
    FrameBuffer: Buffer template for frame data, including timestamp, color, depth, and camera matrices.
"""

from dataclasses import dataclass
import numpy as np

from cyclone.defaults import Config
from cyclone.idl_shared_memory.base_idl import BaseIDL


@dataclass
class FrameIDL(BaseIDL):
    """
    Buffer template for frame data, including timestamp, color, depth, and camera matrices.

    Extends BufferTemplate to define the structure of the data to be shared via shared memory.

    Attributes:
        timestamp (np.ndarray): A 1-element array containing the timestamp (float64).
        color (np.ndarray): An array containing color image data (480x848x3, uint8).
        depth (np.ndarray): An array containing depth image data (480x848, uint16).
        extrinsics (np.ndarray): A 4x4 array representing the extrinsic camera matrix (float64).
        intrinsics (np.ndarray): A 3x3 array representing the intrinsic camera matrix (float64).
    """
    timestamp: np.ndarray = np.empty((1,), dtype=np.float64)
    color: np.ndarray = np.empty((Config.H_d405, Config.W_d405, 3), dtype=np.uint8)
    depth: np.ndarray = np.empty((Config.H_d405, Config.W_d405), dtype=np.uint16)
    extrinsics: np.ndarray = np.empty((4, 4), dtype=np.float64)
    intrinsics: np.ndarray = np.empty((3, 3), dtype=np.float64)
