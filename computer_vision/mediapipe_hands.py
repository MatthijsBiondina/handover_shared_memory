import mediapipe
import numpy as np
import torch
from torch import tensor, Tensor

from cantrips.configs import load_config
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from computer_vision.meshgridcache import MeshgridCache
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl_shared_memory.frame_idl import FrameIDL
from cyclone.idl_shared_memory.mediapipe_idl import MediapipeIDL
from cyclone.idl_shared_memory.points_idl import PointsIDL
from cyclone.patterns.sm_reader import SMReader
from cyclone.patterns.sm_writer import SMWriter


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.points = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_POINTCLOUD,
            idl_dataclass=PointsIDL(),
        )


class Writers:
    def __init__(self, participant: CycloneParticipant):
        self.hand = SMWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.MEDIAPIPE_HAND,
            idl_dataclass=MediapipeIDL(),
        )


logger = get_logger()


class MediapipeHands:
    def __init__(self, domain_participant: CycloneParticipant):
        self.config = load_config()
        self.participant = domain_participant
        self.readers = Readers(domain_participant)
        self.writers = Writers(domain_participant)

        self.mediapipe_hands = mediapipe.solutions.hands
        self.model = self.mediapipe_hands.Hands()

    def run(self):
        while True:
            try:
                frame = self.readers.points()
                centroid_uv, handedness = self.get_hand_uv(frame)
                mask, centroid_xyz = self.get_hand_xyz(frame, centroid_uv)

                msg = MediapipeIDL(
                    timestamp=frame.timestamp,
                    color=frame.color,
                    depth=frame.depth,
                    points=frame.points,
                    extrinsics=frame.extrinsics,
                    intrinsics=frame.intrinsics,
                    mask=mask.cpu().numpy().astype(np.bool_),
                    centroid_uv=centroid_uv.cpu().numpy().astype(np.int32),
                    centroid_xyz=centroid_xyz.cpu().numpy().astype(np.float32),
                    right_handedness=np.array([handedness == "Right"])
                )
                self.writers.hand(msg)
            except ContinueException:
                pass
            finally:
                self.participant.sleep()

    def get_hand_uv(self, frame: FrameIDL):
        results = self.model.process(frame.color)
        if results.multi_hand_landmarks is None or not results.multi_handedness:
            raise ContinueException

        centroids, handednesses = [], []
        H, W, _ = frame.color.shape
        for ii, hand in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[ii].classification[0].label
            
            if handedness == "Left":# Mediapipe thinks left is right and left is right
                handedness = "Right"
            else:
                handedness = "Left"
            
            # Todo: account for left-handed people
            if not handedness == "Right":
                raise ContinueException
            

            centroid = np.mean(
                [(landmark.x, landmark.y) for landmark in hand.landmark], axis=0
            ) * np.array([W, H])
            centroids.append(centroid)
            handednesses.append(handedness)
        if len(centroids) == 0:
            raise ContinueException

        depth = tensor(
            frame.depth.astype(np.float32),
            dtype=torch.float32,
            device=self.config.device,
        )
        centroid, distance, handedness = None, np.inf, None
        for ii, hand in enumerate(centroids):
            hand = tensor(hand, dtype=torch.float32, device=self.config.device)
            px_distance = torch.linalg.norm(
                MeshgridCache().get_meshgrid((H, W)) - hand[None, None, :], dim=-1
            )

            depth[(depth == 0) | (px_distance > H / 20)] = torch.inf
            if torch.any(~torch.isinf(depth)):
                argmin = torch.argmin(depth)
                u, v = argmin % W, argmin // W

                if depth[v, u] < distance:
                    centroid = tensor(
                        [u, v], dtype=torch.float32, device=self.config.device
                    )
                    distance = depth[v, u]
                    handendness = handednesses[ii]
        if centroid is None:
            raise ContinueException
        return centroid, handedness

    def get_hand_xyz(self, frame: FrameIDL, centroid_uv: Tensor):
        points = tensor(frame.points, dtype=torch.float32, device=self.config.device)
        u, v = centroid_uv.to(torch.int)
        centroid_xyz_closest = tensor(
            points[v, u], dtype=torch.float32, device=self.config.device
        )

        mask = (
            torch.linalg.norm(centroid_xyz_closest - points, dim=-1)
            < self.config.world_distance_threshold
        )
        if not torch.any(mask):
            raise ContinueException
        centroid_xyz = torch.nanmedian(points[mask], dim=0).values

        return mask, centroid_xyz


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = MediapipeHands(participant)
    with torch.no_grad():
        node.run()
