import numpy as np
import torch
from torch import tensor
from torch.cuda import device

from cantrips.configs import load_config
from cantrips.debugging.terminal import pyout
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

        logger.warning(f"PointClouds: Ready!")

    def run(self):
        while True:
            frame = self.readers.d405()
            pointcloud = self.back_project(
                frame.depth, frame.intrinsics, frame.extrinsics
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
        pixel_radius: int = 10,
        xyz_radius: float = 0.10,
    ):
        grid = MeshgridCache().get_meshgrid(depth.shape)
        uv = tensor(uv, device=CONFIG.device, dtype=torch.float32)
        bs = uv.shape[0]

        D_px = torch.linalg.norm(grid[None, :, :, :] - uv[:, None, None, :], dim=-1)
        px_mask = D_px < pixel_radius

        D = tensor(
            depth * CONFIG.depth_scale, device=CONFIG.device, dtype=torch.float32
        )
        D[(D == 0) | (D > CONFIG.event_horizon)] = torch.nan
        D = D[None, ...].repeat(bs, 1, 1)
        D[torch.isnan(D) | ~px_mask] = torch.inf
        argmin = torch.argmin(D.view(bs, -1), dim=-1)

        points = tensor(pointcloud, device=CONFIG.device, dtype=torch.float32)
        obj_points = points.view(-1, 3)[argmin]
        obj_points = obj_points[~torch.any(torch.isnan(obj_points), dim=-1)]

        D_xyz = torch.linalg.norm(points[None, ...] - obj_points[:, None, None, :], dim=-1)
        xyz_mask = (D_xyz < xyz_radius).view(obj_points.shape[0], -1)
        points = points.view(-1, 3)[None, ...].repeat(obj_points.shape[0], 1, 1)
        points[~xyz_mask] = torch.nan

        obj_xyz, _ = torch.nanmedian(points, dim=1)

        return obj_xyz.cpu().numpy()


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = PointClouds(participant)
    node.run()
