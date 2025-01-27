from typing import Tuple, List

import mediapipe as mp
import numpy as np
import torch
from cyclonedds.domain import DomainParticipant
from cyclonedds.util import timestamp
from torch import tensor

from cantrips.configs import load_config
from cantrips.debugging.terminal import pyout
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from computer_vision.meshgridcache import MeshgridCache
from computer_vision.pointclouds import PointClouds
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl_shared_memory.frame_idl import FrameIDL
from cyclone.idl_shared_memory.mediapipe_idl import MediapipeIDL
from cyclone.idl_shared_memory.points_idl import PointsIDL
from cyclone.patterns.sm_reader import SMReader
from cyclone.patterns.sm_writer import SMWriter
from utils.image_processing_utils import make_pixel_grid

logger = get_logger()


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.points = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_POINTCLOUD,
            idl_dataclass=PointsIDL(),
        )


class Writers:
    def __init__(self, participant: CycloneParticipant):
        self.pose = SMWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.MEDIAPIPE_POSE,
            idl_dataclass=MediapipeIDL(),
        )


class MediapipePose:
    HAND_LANDMARK_INDICES = [15, 16, 17, 18, 19, 20, 21, 22]
    HAND_INDICES = {
        "Left": {
            "left_wrist": 15,
            "left_thumb": 17,
            "left_index": 19,
            "left_pinky": 21,
        },
        "Right": {
            "right_wrist": 16,
            "right_thumb": 18,
            "right_index": 20,
            "right_pinky": 22,
        },
    }
    VISIBILITY_THRESHOLD = 0.8
    WORLD_DISTANCE_THRESHOLD = 0.1

    def __init__(self, domain_participant: CycloneParticipant):
        self.config = load_config()
        self.participant = domain_participant
        self.readers = Readers(domain_participant)
        self.writers = Writers(domain_participant)

        self.mediapipe_pose = mp.solutions.pose
        self.model = self.mediapipe_pose.Pose()
        logger.info("Mediapipe: Ready!")

    def run(self):
        while True:
            try:
                frame: PointsIDL = self.readers.points()
                results = self.model.process(frame.color)
                if not results.pose_landmarks:
                    continue
                # landmarks = self.get_landmarks(results, frame.color.shape)

                try:
                    uv_ = self.get_hand_landmarks(results, frame.color.shape)
                    xyz_ = PointClouds.uv2xyz(uv_, frame.depth, frame.points)
                except IndexError:
                    uv_ = np.empty((0, 2))
                    xyz_ = np.empty((0, 3))

                uv = np.full((8, 2), np.nan, dtype=np.float32)
                xyz = np.full((8,3), np.nan, dtype=np.float32)

                uv[:uv_.shape[0]] = uv_
                xyz[:xyz_.shape[0]] = xyz_

                msg = MediapipeIDL(
                    timestamp=frame.timestamp,
                    color = frame.color,
                    depth=frame.depth,
                    points=frame.points,
                    extrinsics=frame.extrinsics,
                    intrinsics=frame.intrinsics,
                    uv=uv,
                    xyz=xyz
                )
                self.writers.pose(msg)
            except ContinueException:
                pass
            finally:
                self.participant.sleep()

    def get_hand_landmarks(self, results, img_shape):
        h, w, _ = img_shape
        landmarks = []
        for _, indices in self.HAND_INDICES.items():
            for idx in indices.values():
                landmark = results.pose_landmarks.landmark[idx]
                if landmark.visibility > self.VISIBILITY_THRESHOLD:
                    landmarks.append([landmark.x * w, landmark.y * h])
        return np.array(landmarks)

    # def get_landmarks(self, results, img_shape):
    #     h, w, _ = img_shape
    # 
    #     uv_landmarks = []
    #     for landmark in results.pose_landmarks.landmark:
    #         uv_landmarks.append([landmark.x * w, landmark.y * h])
    # 
    #     return np.array(uv_landmarks)


    # def get_hands(self, results, img_shape):
    #     h, w, _ = img_shape
    # 
    #     hands, handednesses = [], []
    #     for handedness, hand_landmarks in self.HAND_INDICES.items():
    #         hand = []
    #         for idx in hand_landmarks.values():
    #             landmark = results.pose_landmarks.landmark[idx]
    #             if landmark.visibility > self.VISIBILITY_THRESHOLD:
    #                 hand.append([landmark.x * w, landmark.y * h])
    #         hands.append(np.mean(hand, axis=0) if len(hand) > 0 else None)
    #         handednesses.append(handedness)
    # 
    #     return hands, handednesses

    # def get_closest_hand_centroid(
    #     self,
    #     hands: List[Tuple[float, float] | None],
    #     handednesses: List[str],
    #     depth_: np.ndarray,
    # ):
    #     depth = (
    #         torch.tensor(
    #             depth_.astype(np.float32),
    #             dtype=torch.float32,
    #             device=self.config.device,
    #         )
    #         * self.config.depth_scale
    #     )
    #     H, W = depth.shape
    #     hands = [hand for hand in hands if hand is not None]
    #     if not len(hands):
    #         raise ContinueException
    # 
    #     centroid, distance, handedness = None, np.inf, None
    #     for ii, hand in enumerate(hands):
    #         hand = torch.tensor(hand, dtype=torch.float32, device=self.config.device)
    # 
    #         pixel_distance = torch.linalg.norm(
    #             MeshgridCache().get_meshgrid(depth.shape) - hand[None, None, :],
    #             dim=-1,
    #         )
    #         depth[(depth == 0) | (pixel_distance > H / 20)] = torch.inf
    # 
    #         if torch.any(~torch.isinf(depth)):
    #             argmin = torch.argmin(depth)
    #             u, v = argmin % depth.shape[1], argmin // depth.shape[1]
    # 
    #             if depth[v, u] < distance:
    #                 centroid = (u, v)
    #                 distance = depth[v, u]
    #                 handedness = handednesses[ii]
    # 
    #     if centroid is None:
    #         raise ContinueException
    #     else:
    #         return tensor(centroid, device=self.config.device), handedness
    # 
    # def calculate_centroid_xyz(self, centroid_uv: Tuple[int, int], points_: np.ndarray):
    #     points = tensor(points_, dtype=torch.float32, device=self.config.device)
    #     u, v = centroid_uv
    #     hand_point_closest = tensor(
    #         points[v, u], dtype=torch.float32, device=self.config.device
    #     )
    # 
    #     mask = (
    #         torch.linalg.norm(hand_point_closest[None, None, :] - points, axis=-1)
    #         < self.config.world_distance_threshold
    #     )
    #     if not torch.any(mask):
    #         raise ContinueException
    #     centroid_xyz = torch.nanmedian(points[mask], axis=0).values
    # 
    #     return mask, centroid_xyz


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = MediapipePose(participant)
    node.run()
