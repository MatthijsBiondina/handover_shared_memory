import cv2
import numpy as np
from cantrips.configs import load_config
from cantrips.debugging.terminal import UGENT, hex2rgb
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from computer_vision.pointclouds import PointClouds
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.procedures.coordinate_sample import CoordinateSample
from cyclone.idl.sensor_fusion.kalman_sample import KalmanSample
from cyclone.idl_shared_memory.frame_idl import FrameIDL
from cyclone.idl_shared_memory.masks_idl import MasksIDL
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.sm_reader import SMReader
from visualization.webimagestreamer import WebImageStreamer

logger = get_logger()


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.sam = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.SAM_MASKS,
            idl_dataclass=MasksIDL(),
        )
        self.frame = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_FRAME,
            idl_dataclass=FrameIDL(),
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


class SAMPlotter:
    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()
        self.participant = participant
        self.readers = Readers(participant)
        self.web_streamer = WebImageStreamer(title="SAM", port=5000)
        logger.info("SAMPlotter: Ready!")

    def run(self):
        while True:
            img = np.zeros((848, 480, 3), dtype=np.uint8)
            try:
                frame: FrameIDL = self.readers.frame()
                if frame is None:
                    raise ContinueException
                img = frame.color

                masks: MasksIDL = self.readers.sam()
                if masks is not None:
                    img = self.overlay_masks(img, masks)

                target: CoordinateSample = self.readers.target()
                if target is not None:
                    img = self.draw_target(img, frame, target)

                hands: KalmanSample = self.readers.hands()
                if hands is not None:
                    img = self.draw_hands(img, frame, hands)
            except ContinueException:
                pass
            finally:
                self.web_streamer.update_frame(img[..., ::-1])
                self.participant.sleep()

    def overlay_masks(self, img: np.ndarray, masks: MasksIDL) -> np.ndarray:
        img[masks.mask_object] = (
            0.5 * img[masks.mask_object] + 0.5 * np.array(hex2rgb(UGENT.BLUE))
        ).astype(np.uint8)
        img[masks.mask_hand] = (
            0.5 * img[masks.mask_hand] + 0.5 * np.array(hex2rgb(UGENT.PINK))
        ).astype(np.uint8)
        return img

    def draw_target(self, img: np.ndarray, frame: FrameIDL, target: CoordinateSample):
        obj_xyz = np.array([[target.x, target.y, target.z]])
        obj_uv = PointClouds.xyz2uv(obj_xyz, frame.intrinsics, frame.extrinsics)

        u, v = obj_uv[0, 0], obj_uv[0, 1]
        img = cv2.circle(
            img, (u, v), radius=5, color=hex2rgb(UGENT.PURPLE), thickness=-1
        )
        return img

    def draw_hands(self, img: np.ndarray, frame: FrameIDL, hands: KalmanSample):
        for mu, Sigma in zip(hands.mean, hands.covariance):
            if np.all(np.diag(np.array(Sigma)) < 0.2**2):
                uv = PointClouds.xyz2uv(np.array(mu)[None, :], frame.intrinsics, frame.extrinsics)
                u, v = uv[0, 0], uv[0, 1]
                img = cv2.circle(
                    img, (u, v), radius=5, color=hex2rgb(UGENT.RED), thickness=-1
                )

        return img


if __name__ == "__main__":
    participant: CycloneParticipant = CycloneParticipant()
    node = SAMPlotter(participant)
    node.run()
