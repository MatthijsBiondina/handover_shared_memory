from typing import Union, Tuple, List

import numpy as np
import torch
from curobo.types.math import Pose
from curobo.types.state import JointState
from scipy.spatial.transform import Rotation as R
from torch import tensor

from cantrips.debugging.terminal import pyout


def numpy2cspace(joints: np.ndarray, joint_names: List[str]):
    if len(joints.shape) == 1:
        joints = joints[None, :]

    js = JointState.from_position(
        tensor(joints, dtype=torch.float32).cuda(),
        joint_names=joint_names
    )
    return js


def matrix2cupose(
    matrix: np.ndarray, gpu=True
) -> Union[Pose, Tuple[np.ndarray, np.ndarray]]:
    # Ensure the input is a 4x4 matrix
    assert matrix.shape == (4, 4), "Input must be a 4x4 matrix"

    # Extract the position (translation part) from the matrix
    position = matrix[:3, 3]

    # Extract the rotation matrix
    rotation_matrix = matrix[:3, :3]

    # Convert the rotation matrix to a quaternion using scipy
    rotation = R.from_matrix(rotation_matrix)
    quaternion_xyzw = rotation.as_quat()  # Returns (x, y, z, w) quaternion

    # Convert position and quaternion into curobo Pose
    quaternion_wxyz = quaternion_xyzw[[-1, 0, 1, 2]]
    if gpu:
        pose = Pose(
            position=tensor(position, dtype=torch.float32).cuda(),
            quaternion=tensor(quaternion_wxyz, dtype=torch.float32).cuda(),
        )
        return pose
    else:
        return position, quaternion_wxyz


def cupose2matrix(pose: Pose) -> np.ndarray:
    # Convert curobo Pose to position and quaternion
    position = pose.position.squeeze(0).cpu().numpy()
    quaternion_wxyz = pose.quaternion.squeeze(0).cpu().numpy()

    # Convert quaternion from wxyz to xyzw format
    quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]

    # Convert quaternion to a 3x3 rotation matrix using scipy
    rotation = R.from_quat(quaternion_xyzw)
    rotation_matrix = rotation.as_matrix()

    # Build the 4x4 homogeneous transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = position

    return matrix


def matrix2euler(matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    # Ensure the input is a 4x4 matrix
    assert matrix.shape == (4, 4), "Input must be a 4x4 matrix"

    # Extract the position
    position = matrix[:3, 3]

    # Extract the rotation matrix
    rotation_matrix = matrix[:3, :3]

    # Convert the rotation matrix to euler using scipy
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler("xyz", degrees=False)

    return position, euler_angles


def euler2matrix(position: np.ndarray, euler_angles: np.ndarray) -> np.ndarray:
    rotation = R.from_euler("xyz", euler_angles, degrees=False)
    rotation_matrix = rotation.as_matrix()

    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = position
    return matrix
