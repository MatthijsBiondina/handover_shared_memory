from typing import final

import numpy as np

from cantrips.configs import load_config
from cantrips.debugging.terminal import hex2rgb, UGENT
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
            idl_dataclass=FrameIDL()
        )
        self.mediapipe_hand = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.MEDIAPIPE_HAND,
            idl_dataclass=MediapipeIDL()
        )

class MediapipeHandPlotter:
    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()

        self.participant = participant
        self.readers = Readers(participant)

        self.web_streamer = WebImageStreamer(title="Mediapipe Hand", port=5003)

    def run(self):
        while True:
            try:
                frame: FrameIDL = self.readers.frame()
                hands: MediapipeIDL = self.readers.mediapipe_hand()

                pointcloud = hands.points
                pointcloud[~hands.mask] = np.nan

                mask = PointClouds.forward_project(
                    pointcloud, frame.intrinsics, frame.extrinsics
                )

                img = frame.color
                img[mask] = np.array(hex2rgb(UGENT.BLUE))

                self.web_streamer.update_frame(img[..., ::-1])
            except ContinueException:
                pass
            finally:
                self.participant.sleep()

if __name__ == "__main__":
    participant = CycloneParticipant()
    node = MediapipeHandPlotter(participant)
    node.run()