import numpy as np
from cantrips.configs import load_config
from cantrips.debugging.terminal import UGENT, hex2rgb
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl_shared_memory.frame_idl import FrameIDL
from cyclone.idl_shared_memory.masks_idl import MasksIDL
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


class SAMPlotter:
    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()
        self.participant = participant
        self.readers = Readers(participant)
        self.web_streamer = WebImageStreamer(title="SAM", port=5006)
        logger.info("SAMPlotter: Ready!")

    def run(self):
        while True:
            try:
                frame: FrameIDL = self.readers.frame()
                masks: MasksIDL = self.readers.sam()
                if frame is None or masks is None:
                    raise ContinueException
                img = self.overlay_masks(frame, masks)
                self.web_streamer.update_frame(img[..., ::-1])
            except ContinueException:
                pass
            finally:
                self.participant.sleep()

    def overlay_masks(self, frame: FrameIDL, masks: MasksIDL) -> np.ndarray:
        img = frame.color
        img[masks.mask_object] = hex2rgb(UGENT.BLUE)
        img[masks.mask_hand] = hex2rgb(UGENT.PINK)
        return img


if __name__ == "__main__":
    participant: CycloneParticipant = CycloneParticipant()
    node = SAMPlotter(participant)
    node.run()
