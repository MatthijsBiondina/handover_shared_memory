from typing import Dict, Type

import numpy as np
from pydrake.geometry import Box
from pydrake.math import RigidTransform

from cyclone.idl.simulation.world_config_rpc_sample import WorldConfigRPC


class DrakeWorldConfig:
    def __init__(self):
        self.mesh: Dict[str, Dict[str, np.ndarray]] = {}
        self.cuboid: Dict[str, Dict[str, np.ndarray]] = {}
        self.capsule: Dict[str, Dict[str, np.ndarray]] = {}
        self.cylinder: Dict[str, Dict[str, np.ndarray]] = {}
        self.sphere: Dict[str, Dict[str, np.ndarray]] = {}

    def add_mesh(self, name, *args):
        raise NotImplemented

    def add_cuboid(self, box: Box, name: str, pose: RigidTransform):
        dimensions = box.size()
        position = pose.translation()
        quaternion = pose.rotation().ToQuaternion()
        self.cuboid[name] = {
            "dims": dimensions,
            "pose": np.array(
                [
                    *position.tolist(),
                    quaternion.w(),
                    quaternion.x(),
                    quaternion.y(),
                    quaternion.z(),
                ]
            ),
        }

    def add_sphere(self, *args):
        raise NotImplemented

    def as_dictionary(self) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        return {
            "mesh": self.mesh,
            "cuboid": self.cuboid,
            "capsule": self.capsule,
            "cylinder": self.cylinder,
            "sphere": self.sphere,
        }

    @staticmethod
    def from_cyclonedds_response(response: WorldConfigRPC.Response):
        world_config = DrakeWorldConfig()

        for name, dims, pose in zip(
            response.cuboid, response.cuboid_dims, response.cuboid_pose
        ):
            world_config.cuboid[name] = {
                "dims": np.array(dims),
                "pose": np.array(pose),
            }

        return world_config
