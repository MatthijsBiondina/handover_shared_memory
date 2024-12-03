import cv2
import numpy as np

from cantrips.configs import load_config
from cantrips.debugging.terminal import pyout, hex2rgb, UGENT
from cantrips.exceptions import ContinueException
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl_shared_memory.yolo_idl import YOLOIDL
from cyclone.patterns.sm_reader import SMReader
from visualization.webimagestreamer import WebImageStreamer


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.yolo = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.YOLO,
            idl_dataclass=YOLOIDL(),
        )


class YOLOPlotter:
    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()

        self.participant = participant
        self.readers = Readers(participant)

        self.web_streamer = WebImageStreamer(title="YOLO", port=5003)

    def run(self):
        while True:
            try:
                yolo: YOLOIDL = self.readers.yolo()
                img = self.draw_boxes(yolo.color, yolo.objects)

                self.web_streamer.update_frame(img[..., ::-1])

            except ContinueException:
                pass
            finally:
                self.participant.sleep()

    def draw_boxes(self, img: np.ndarray, objects: np.ndarray):
        for obj in objects:
            if np.any(np.isnan(obj)):
                continue
            img = cv2.rectangle(img,
                                (int(obj[0]), int(obj[1])),
                                (int(obj[2]), int(obj[3])),
                                color=hex2rgb(UGENT.BLUE),
                                thickness=3)
        return img



if __name__ == "__main__":
    participant = CycloneParticipant()
    node = YOLOPlotter(participant)
    node.run()
