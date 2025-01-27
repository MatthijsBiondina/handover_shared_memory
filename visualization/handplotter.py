import cv2
import numpy as np
from cantrips.configs import load_config
from cantrips.debugging.terminal import UGENT, hex2rgb
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from computer_vision.mediapipe_hands import HAND_CONNECTIONS
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.mediapipe.hand_sample import HandSample
from cyclone.idl_shared_memory.frame_idl import FrameIDL
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.sm_reader import SMReader
from visualization.webimagestreamer import WebImageStreamer


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.frame = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_FRAME,
            idl_dataclass=FrameIDL(),
        )
        self.hand = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.MEDIAPIPE_HAND,
            idl_dataclass=HandSample,
        )


logger = get_logger()


class HandPlotter:
    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()
        self.participant = participant
        self.readers = Readers(participant)
        self.web_streamer = WebImageStreamer(title="Hand Keypoints", port=5008)

    def run(self):
        while True:
            img = np.zeros((848, 480, 3), dtype=np.uint8)
            try:
                frame: FrameIDL = self.readers.frame()
                if frame is None:
                    raise ContinueException
                img = frame.color

                hand: HandSample = self.readers.hand()
                if hand is None:
                    raise ContinueException

                img = self.draw_hand(img, hand)
            except ContinueException:
                pass
            finally:
                self.web_streamer.update_frame(img[...,::-1])
                self.participant.sleep()

    def draw_hand(self, img: np.ndarray, hand: HandSample):
        for kp1, kp2 in HAND_CONNECTIONS:
            coords1 = eval(f"hand.{kp1}")
            coords2 = eval(f"hand.{kp2}")
            if coords1 is None or coords2 is None:
                continue

            u1, v1 = int(coords1[3]), int(coords1[4])
            u2, v2 = int(coords2[3]), int(coords2[4])

            img = cv2.line(img, (u1, v1), (u2, v2), hex2rgb(UGENT.BLUE), thickness=3)

        return img


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = HandPlotter(participant)
    node.run()
