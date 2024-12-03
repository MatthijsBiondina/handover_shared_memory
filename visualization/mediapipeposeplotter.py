import cv2
import numpy as np

from cantrips.configs import load_config
from cantrips.debugging.terminal import UGENT, hex2rgb, pyout
from cantrips.exceptions import ContinueException
from computer_vision.pointclouds import PointClouds
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl_shared_memory.frame_idl import FrameIDL
from cyclone.idl_shared_memory.mediapipe_idl import MediapipeIDL
from cyclone.patterns.sm_reader import SMReader
from visualization.webimagestreamer import WebImageStreamer


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.frame = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_FRAME,
            idl_dataclass=FrameIDL(),
        )
        self.mediapipe_pose = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.MEDIAPIPE_POSE,
            idl_dataclass=MediapipeIDL(),
        )


class MediapipePosePlotter:
    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()

        self.participant = participant
        self.readers = Readers(participant)

        self.web_streamer = WebImageStreamer(title="Mediapipe Pose", port=5002)

    def run(self):
        while True:
            try:
                # frame: FrameIDL = self.readers.frame()
                hands: MediapipeIDL = self.readers.mediapipe_pose()

                pointcloud = hands.points
                pointcloud[~hands.mask] = np.nan

                mask = PointClouds.forward_project(
                    pointcloud, hands.intrinsics, hands.extrinsics
                )

                img = hands.color
                img[mask] = np.array(hex2rgb(UGENT.BLUE))
                img = self.add_landmarks(img, hands.landmarks)

                self.web_streamer.update_frame(img[..., ::-1])

            except ContinueException:
                pass
            finally:
                self.participant.sleep()

    def add_landmarks(self, img: np.ndarray, landmarks: np.ndarray):
        for landmark in landmarks[[16,18,20,22]]:
            img = cv2.circle(img, (int(landmark[0]), int(landmark[1])), radius=5,
                             color=hex2rgb(UGENT.GREEN), thickness=-1)

        return img


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = MediapipePosePlotter(participant)
    node.run()
