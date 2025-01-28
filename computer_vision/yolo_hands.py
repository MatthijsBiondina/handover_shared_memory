import numpy as np
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl_shared_memory.mediapipe_idl import MediapipeIDL
from cyclone.idl_shared_memory.points_idl import PointsIDL
from cyclone.patterns.sm_reader import SMReader
from cyclone.patterns.sm_writer import SMWriter
from ultralytics import YOLO
import mediapipe as mp


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


logger = get_logger()


class YOLOHands:
    MODEL_PATH = "/home/matt/Python/Nature/runs/detect/hands/weights/best.pt"
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

    def __init__(self, participant: CycloneParticipant):
        self.participant = participant
        self.readers = Readers(participant)
        self.writers = Writers(participant)
        self.yolo = YOLO(self.MODEL_PATH)
        self.mp_pose = mp.solutions.pose
        self.mediapipe = self.mp_pose.Pose()

        logger.info("YOLOHands Ready!")

    def run(self):
        while True:
            try:
                points: PointsIDL = self.readers.points()
                if points is None:
                    raise ContinueException
                uv_yolo, xyz_yolo = self.find_hands_with_yolo(points)
                uv_mp, xyz_mp = self.find_hands_with_mediapipe(points)

                uv = np.concatenate((uv_yolo, uv_mp), axis=0)
                xyz = np.concatenate((xyz_yolo, xyz_mp), axis=0)
                if uv.shape[0] > 8:
                    uv = uv[:8]
                    xyz = xyz[:8]

                uv_msg = np.full((8, 2), np.nan, dtype=np.float32)
                xyz_msg = np.full((8, 3), np.nan, dtype=np.float32)
                uv_msg[: uv.shape[0]] = uv
                xyz_msg[: xyz.shape[0]] = xyz

                msg = MediapipeIDL(
                    timestamp=points.timestamp,
                    color=points.color,
                    depth=points.depth,
                    points=points.points,
                    extrinsics=points.extrinsics,
                    intrinsics=points.intrinsics,
                    uv=uv_msg,
                    xyz=xyz_msg,
                )
                self.writers.pose(msg)
            except ContinueException:
                pass
            except ValueError:
                pass
            finally:
                self.participant.sleep()

    def find_hands_with_yolo(self, points: PointsIDL):
        h, w, _ = points.color.shape
        results_top = self.yolo(points.color[:w])
        results_bot = self.yolo(points.color[-w:])

        boxes = []
        if len(results_top):
            for box in results_top[0].boxes:
                boxes.append(
                    {
                        "uvuv": box.xyxy.squeeze(0).cpu().numpy(),
                        "conf": box.conf.cpu().item(),
                    }
                )

        # todo: debug line
        boxes = []

        if len(results_bot):
            for box in results_bot[0].boxes:
                boxes.append(
                    {
                        "uvuv": box.xyxy.squeeze(0).cpu().numpy()
                        + np.array([0, h - w, 0, h - w]),
                        "conf": box.conf.cpu().item(),
                    }
                )
        uv_ = []
        xyz_ = []

        boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)

        for box in boxes:
            uvuv = box["uvuv"]
            u = int(np.clip((uvuv[0] + uvuv[2]) / 2, 0, w - 1))
            v = int(np.clip((uvuv[1] + uvuv[3]) / 2, 0, h - 1))
            uv_.append([u, v])
            xyz_.append(points.points[v, u])

        uv = np.array(uv_, dtype=np.float32)
        xyz = np.array(xyz_, dtype=np.float32)

        if uv.shape[0] > 8:
            uv = uv[:8]
            xyz = xyz[:8]
        elif uv.shape[0] == 0:
            uv = np.empty((0, 2), dtype=np.float32)
            xyz = np.empty((0, 3), dtype=np.float32)

        return uv, xyz

    def find_hands_with_mediapipe(self, points: PointsIDL):
        h, w, _ = points.color.shape
        pad = (h - w) // 2
        img = points.color[pad : w + pad]
        results = self.mediapipe.process(img)
        landmarks = []
        for _, indices in self.HAND_INDICES.items():
            for idx in indices.values():
                if not results.pose_landmarks:
                    continue
                landmark = results.pose_landmarks.landmark[idx]
                if landmark.visibility > self.VISIBILITY_THRESHOLD:
                    landmarks.append([landmark.x * w, landmark.y * w + pad])

        if not len(landmarks):
            return np.empty((0, 2)), np.empty((0, 3))

        uv = np.array(landmarks).astype(int)
        xyz = []
        for u, v in uv:
            try:
                xyz.append(points.points[v, u])
            except IndexError:
                pass
        xyz = np.array(xyz)

        if uv.shape[0] == 0:
            uv = np.empty((0, 2), dtype=np.float32)
            xyz = np.empty((0, 3), dtype=np.float32)
        return uv, xyz


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = YOLOHands(participant)
    node.run()
