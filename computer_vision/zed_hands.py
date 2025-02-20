import time
import traceback
import cv2
import numpy as np
from cantrips.debugging.terminal import UGENT, hex2rgb
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from computer_vision.pointclouds import PointClouds
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.procedures.coordinate_sample import CoordinateSample
from cyclone.idl_shared_memory.mediapipe_idl import MediapipeIDL
from cyclone.idl_shared_memory.zed_idl import ZEDIDL
from cyclone.idl_shared_memory.zed_points_idl import ZedPointsIDL
from cyclone.patterns.ddswriter import DDSWriter
from cyclone.patterns.sm_reader import SMReader
from cyclone.patterns.sm_writer import SMWriter
from ultralytics import YOLO
import mediapipe as mp

from visualization.webimagestreamer import WebImageStreamer


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.frame = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.ZED_FRAME,
            idl_dataclass=ZEDIDL(),
        )


class Writers:
    def __init__(self, participant: CycloneParticipant):
        self.pose = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.ZED_MEDIAPIPE,
            idl_dataclass=CoordinateSample,
        )


logger = get_logger()


class ZedMediapipe:
    VISIBILITY_THRESHOLD = 0.8

    def __init__(self, participant: CycloneParticipant):
        self.participant = participant
        self.readers = Readers(participant)
        self.writers = Writers(participant)
        self.mp_pose = mp.solutions.pose
        self.mediapipe = self.mp_pose.Pose()

        self.webstreamer = WebImageStreamer("Mediapipe", port=5010)

        logger.info("ZedHands Ready!")

    def run(self):
        while True:
            try:
                points: ZEDIDL = self.readers.frame()
                if points is None:
                    raise ContinueException
                self.process_frame(points)
            except ContinueException:
                pass
            finally:
                self.participant.sleep()

    def process_frame(self, points: ZEDIDL):
        person = self.find_hands_with_mediapipe(points)

        if person is None:
            return

        msg = CoordinateSample(
            timestamp=time.time(), x=person[0], y=person[1], z=person[2]
        )
        self.writers.pose(msg)

    def find_hands_with_mediapipe(self, points: ZEDIDL):
        h, w, _ = points.color.shape
        img = points.color
        results = self.mediapipe.process(img)
        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark
                U = np.array([l.x * w for l in landmarks]).astype(int)
                V = np.array([l.y * h for l in landmarks]).astype(int)

                mask = (U > 0) & (U < w) & (V > 0) & (V < h)

                U = U[mask]
                V = V[mask]
                if U.shape[0] == 0:
                    return None

                for u, v in zip(U, V):
                    img = cv2.circle(img, (u, v), 5, hex2rgb(UGENT.GREEN), -1)
                self.webstreamer.update_frame(img)

                XYZ = PointClouds.back_project(
                    points.depth, points.intrinsics, points.extrinsics, depth_scale=1
                )[V, U]
                return np.nanmedian(XYZ, axis=0)
            except Exception as e:
                logger.info(traceback.format_exc())
                return None
        return None


if __name__ == "__main__":
    participant = CycloneParticipant(rate_hz=10)
    node = ZedMediapipe(participant)
    node.run()
