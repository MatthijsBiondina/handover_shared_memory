import os
from pathlib import Path

import cv2
import numpy as np
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl_shared_memory.frame_idl import FrameIDL
from cyclone.patterns.sm_reader import SMReader


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.d405 = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_FRAME,
            idl_dataclass=FrameIDL(),
        )


logger = get_logger()


class RS2Recorder:
    ROOT = Path("/home/matt/Videos")

    def __init__(self, participant: CycloneParticipant):
        self.participant = participant
        self.readers = Readers(participant)

        self.timestamp = None
        self.root = None
        self.paused = True

        logger.info("RS2-Recorder Ready!")

    def __init_savedir(self):
        ii = 0
        while True:
            folder = self.ROOT / f"rs2_{str(ii).zfill(2)}"
            if os.path.isdir(folder):
                ii += 1
            else:
                os.makedirs(folder)
                return folder

    def run(self):
        while True:
            try:
                frame: FrameIDL = self.readers.d405()
                if frame is None:
                    raise ContinueException
                if (
                    self.timestamp is not None
                    and frame.timestamp.item() <= self.timestamp
                ):
                    raise ContinueException
                self.timestamp = frame.timestamp.item()

                if self.paused:
                    img = cv2.cvtColor(frame.color[..., ::-1], cv2.COLOR_BGR2GRAY)
                    img = np.repeat(img[..., None], 3, axis=-1)
                    cv2.imshow("RS2 Recorder", img)
                    if cv2.waitKey(1) & 0xFF == ord("r"):
                        self.root = self.__init_savedir()
                        self.paused = False
                else:
                    cv2.imwrite(
                        str(self.root / f"{self.timestamp:.2f}.jpg"),
                        frame.color[..., ::-1],
                    )
                    cv2.imshow("RS2 Recorder", frame.color[..., ::-1])
                    if cv2.waitKey(1) & 0xFF == ord("p"):
                        self.paused = True

            except ContinueException:
                pass
            finally:
                self.participant.sleep()


if __name__ == "__main__":
    participant = CycloneParticipant(rate_hz=30)
    node = RS2Recorder(participant)
    node.run()
