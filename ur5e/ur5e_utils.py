import numpy as np
from cantrips.logging.logger import get_logger


logger = get_logger()


def look_at(position: np.ndarray, focus: np.ndarray):
    z_axis = focus - position
    z_axis = z_axis / np.linalg.norm(z_axis)

    x_axis = np.cross(np.array([0.0, 0.0, 1.0]), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    R = np.array([x_axis, y_axis, z_axis]).T

    tcp = np.eye(4)
    tcp[:3, :3] = R
    tcp[:3, 3] = position

    return tcp


def compute_approach_pose(x: float, y: float, z: float, distance=0.2) -> np.ndarray:
    focus_point = np.array([x, y, z])
    approach_point = focus_point.copy()
    approach_point[:2] -= distance * focus_point[:2] / np.linalg.norm(focus_point[:2])

    return look_at(approach_point, focus_point)
