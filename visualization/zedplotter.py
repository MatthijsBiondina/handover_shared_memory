import time
from airo_camera_toolkit.calibration.fiducial_markers import (
    detect_and_visualize_charuco_pose,
)
import cv2
from cyclonedds.domain import DomainParticipant
import numpy as np
from cantrips.configs import load_config
from cantrips.debugging.terminal import UGENT, hex2rgb
from cantrips.exceptions import WaitingForFirstMessageException
from cantrips.logging.logger import get_logger
from computer_vision.pointclouds import PointClouds
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.procedures.coordinate_sample import CoordinateSample
from cyclone.idl.sensor_fusion.kalman_sample import KalmanSample
from cyclone.idl_shared_memory.zed_hands_idl import ZedHandsIDL
from cyclone.idl_shared_memory.zed_idl import ZEDIDL
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.sm_reader import SMReader
from visualization.webimagestreamer import WebImageStreamer

logger = get_logger()


class Readers:
    def __init__(self, participant: DomainParticipant):
        self.frame = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.ZED_FRAME,
            idl_dataclass=ZEDIDL(),
        )
        self.target = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.TARGET_OBJECT,
            idl_dataclass=CoordinateSample,
        )
        self.hands = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.KALMAN_HANDS,
            idl_dataclass=KalmanSample,
        )
        self.person = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.ZED_MEDIAPIPE,
            idl_dataclass=CoordinateSample,
        )


class ZEDPlotter:
    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()
        self.participant = participant
        self.readers = Readers(participant)
        # Create an instance of WebImageStreamer
        self.web_streamer = WebImageStreamer(title="ZED", port=5001)

    def run(self):
        while True:
            try:
                frame: ZEDIDL = self.readers.frame()
                if frame is None:
                    raise WaitingForFirstMessageException
                img = frame.color
                img = self.draw_charuco(img, frame.intrinsics)

                target: CoordinateSample = self.readers.target()
                if target is not None:
                    img = self.draw_keypoint(img, frame, target)

                person: CoordinateSample = self.readers.person()
                if person is not None:
                    img = self.draw_keypoint(img, frame, person, color=UGENT.GREEN)

                hands: KalmanSample = self.readers.hands()
                if hands is not None:
                    img = self.draw_hands(img, frame, hands)

                self.web_streamer.update_frame(img)

            except WaitingForFirstMessageException:
                pass
            finally:
                self.participant.sleep()

    def draw_charuco(self, img: np.ndarray, intrinsics: np.ndarray):
        detect_and_visualize_charuco_pose(img, intrinsics)
        return img

    def draw_keypoint(self, img: np.ndarray, frame: ZEDIDL, keypoint: CoordinateSample, color=UGENT.PURPLE):
        obj_xyz = np.array([[keypoint.x, keypoint.y, keypoint.z]])
        obj_uv = PointClouds.xyz2uv(obj_xyz, frame.intrinsics, frame.extrinsics)

        u, v = obj_uv[0, 0], obj_uv[0, 1]
        img = cv2.circle(
            img, (u, v), radius=5, color=hex2rgb(color), thickness=-1
        )
        return img

    def draw_hands(self, img: np.ndarray, frame: ZEDIDL, hands: KalmanSample):
        for mu, Sigma in zip(hands.mean, hands.covariance):
            if np.all(np.diag(np.array(Sigma)) < 0.2**2):
                uv = PointClouds.xyz2uv(
                    np.array(mu)[None, :], frame.intrinsics, frame.extrinsics
                )
                u, v = uv[0, 0], uv[0, 1]
                img = cv2.circle(
                    img, (u, v), radius=5, color=hex2rgb(UGENT.RED), thickness=-1
                )

        return img


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = ZEDPlotter(participant)
    node.run()
