import time
import cv2
import numpy as np
import torch
from torch import tensor
from torch.cuda import device

from cantrips.configs import load_config
from cantrips.debugging.terminal import UGENT, hex2rgb, pyout
from cantrips.logging.logger import get_logger
from computer_vision.meshgridcache import MeshgridCache
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl_shared_memory.frame_idl import FrameIDL
from cyclone.idl_shared_memory.points_idl import PointsIDL
from cyclone.patterns.sm_reader import SMReader
from cyclone.patterns.sm_writer import SMWriter


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.d405 = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_FRAME,
            idl_dataclass=FrameIDL(),
        )


class Writers:
    def __init__(self, participant: CycloneParticipant):
        self.points = SMWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_POINTCLOUD,
            idl_dataclass=PointsIDL(),
        )


logger = get_logger()
CONFIG = load_config()


class PointClouds:
    def __init__(self, participant: CycloneParticipant):
        self.participant = participant
        self.readers = Readers(participant)
        self.writers = Writers(participant)

        logger.info(f"PointClouds: Ready!")

    def run(self):
        while True:
            frame = self.readers.d405()
            masked_depth = self.apply_finger_mask(frame)
            pointcloud = self.back_project(
                masked_depth, frame.intrinsics, frame.extrinsics
            )
            self.writers.points(
                PointsIDL(
                    timestamp=frame.timestamp,
                    color=frame.color,
                    depth=frame.depth,
                    points=pointcloud,
                    extrinsics=frame.extrinsics,
                    intrinsics=frame.intrinsics,
                )
            )

            self.participant.sleep()

    def apply_finger_mask(self, frame: FrameIDL):
        depth = frame.depth
        depth[:200, :50] = 0
        depth[:200, -50:] = 0
        return depth

    @staticmethod
    def back_project(
        depth_: np.ndarray, intrinsics_: np.ndarray, extrinsics_: np.ndarray
    ):
        """
        Converts a depth image to a point cloud in the world frame.
        :param depth:
        :param intrinsics:
        :param extrinsics:
        :return:
        """
        # preprocess depth image
        with torch.no_grad():
            depth = tensor(
                depth_.astype(np.float32), device=CONFIG.device, dtype=torch.float32
            )
            intrinsics = tensor(intrinsics_, device=CONFIG.device, dtype=torch.float32)
            extrinsics = tensor(extrinsics_, device=CONFIG.device, dtype=torch.float32)

            H, W = depth.shape
            depth = depth * CONFIG.depth_scale
            depth[(depth == 0) | (depth > CONFIG.event_horizon)] = np.nan

            meshgrid = MeshgridCache().get_meshgrid((H, W))
            u_grid, v_grid = meshgrid[..., 0], meshgrid[..., 1]
            u_flat, v_flat = u_grid.flatten(), v_grid.flatten()
            depth_flat = depth.flatten()

            # Intrinsics parameters
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]

            # Back-project pixels to camera coordinates
            x = (u_flat - cx) * depth_flat / fx
            y = (v_flat - cy) * depth_flat / fy
            z = depth_flat

            points_camera = torch.vstack((x, y, z, torch.ones_like(z)))

            # Transform to world coordinates
            points_world = (extrinsics @ points_camera)[:3].T
            points_world = points_world.reshape((H, W, 3))

            return points_world.cpu().numpy()

    @staticmethod
    def forward_project(
        pointcloud_: np.ndarray, intrinsics_: np.ndarray, extrinsics_: np.ndarray
    ):
        """
        Converts a point cloud to a mask using camera intrinsics and extrinsics
        :param pointcloud_:
        :param intrinsics_:
        :param extrinsics_:
        :return:
        """
        with torch.no_grad():
            pointcloud = tensor(
                pointcloud_.astype(np.float32),
                dtype=torch.float32,
                device=CONFIG.device,
            )
            H, W = pointcloud.shape[:2]

            pointcloud = pointcloud.view(-1, 3)
            intrinsics = tensor(intrinsics_, device=CONFIG.device, dtype=torch.float32)
            extrinsics = tensor(extrinsics_, device=CONFIG.device, dtype=torch.float32)

            # Transform points to camera coordinates
            extrinsics_inv = torch.linalg.inv(extrinsics)
            points_world_homogeneous = torch.hstack(
                (pointcloud, torch.ones((pointcloud.shape[0], 1), device=CONFIG.device))
            )
            points_camera = (extrinsics_inv @ points_world_homogeneous.T).T[:, :3]

            # Project onto image plane
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            u = ((points_camera[:, 0] * fx / points_camera[:, 2]) + cx).to(torch.int)
            v = ((points_camera[:, 1] * fy / points_camera[:, 2]) + cy).to(torch.int)

            valid_indices = (
                (u >= 0) & (u < W) & (v >= 0) & (v < H) & (points_camera[:, 2] > 0)
            )
            mask = valid_indices.view((H, W))

            return mask.cpu().numpy()

    @staticmethod
    def uv2xyz(
        uv: np.ndarray,
        depth: np.ndarray,
        pointcloud: np.ndarray,
        pixel_radius: int = 2,
        xyz_radius: float = 0.1,
    ):
        u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
        xyz = pointcloud[v, u]
        return xyz

    @staticmethod
    def xyz2uv(xyz: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray):
        pointcloud = tensor(xyz, device=CONFIG.device, dtype=torch.float32)
        intrinsics = tensor(intrinsics, device=CONFIG.device, dtype=torch.float32)
        extrinsics = tensor(extrinsics, device=CONFIG.device, dtype=torch.float32)

        extrinsics_inv = torch.linalg.inv(extrinsics)
        points_world_homogeneous = torch.hstack(
            (pointcloud, torch.ones((pointcloud.shape[0], 1), device=CONFIG.device))
        )
        points_camera = (extrinsics_inv @ points_world_homogeneous.T).T[:, :3]

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        u = ((points_camera[:, 0] * fx / points_camera[:, 2]) + cx).to(torch.int)
        v = ((points_camera[:, 1] * fy / points_camera[:, 2]) + cy).to(torch.int)

        uv = torch.stack((u, v), dim=-1)
        return uv.cpu().numpy()

        pyout()


if __name__ == "__main__":
    with torch.no_grad():
        participant = CycloneParticipant()
        node = PointClouds(participant)
        node.run()
