import json
import logging
import os

import numpy as np

from cantrips.debugging.terminal import pyout
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl_shared_memory.frame_idl import FrameIDL
from cyclone.idl_shared_memory.mediapipe_idl import MediapipeIDL
from cyclone.idl_shared_memory.points_idl import PointsIDL
from cyclone.idl_shared_memory.yolo_idl import YOLOIDL
from cyclone.patterns.sm_reader import SMReader
from ultralytics import YOLO
import ultralytics


from cyclone.patterns.sm_writer import SMWriter

logger = get_logger()
logging.getLogger("ultralytics").setLevel(logging.ERROR)


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.frame = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_POINTCLOUD,
            idl_dataclass=PointsIDL(),
        )


class Writers:
    def __init__(self, participant: CycloneParticipant):
        self.yolo = SMWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.YOLO,
            idl_dataclass=YOLOIDL(),
        )


class YOLOLabeler:
    def __init__(self, participant: CycloneParticipant):
        self.participant = participant
        self.readers = Readers(participant)
        self.writers = Writers(participant)
        self.MAX_NR_OF_OBJECTS = YOLOIDL().objects.shape[0]

        self.yolo_model = YOLO("yolov8x")


        with open(f"{os.path.dirname(__file__)}/../config/yolo_names.json", "w+") as f:
            json.dump(self.yolo_model.names, f, indent=2)

        logger.warning("YOLOLabeler: Ready!")

    def run(self):
        while True:
            try:
                frame: PointsIDL = self.readers.frame()
                results = self.yolo_model(frame.color)
                if not len(results):
                    raise ContinueException

                objects_array = self.extract_objects(results[0])

                msg = YOLOIDL(
                    timestamp=frame.timestamp,
                    color=frame.color,
                    depth=frame.depth,
                    points=frame.points,
                    extrinsics=frame.extrinsics,
                    intrinsics=frame.intrinsics,
                    objects=objects_array,
                )
                self.writers.yolo(msg)
            except ContinueException:
                pass
            finally:
                self.participant.sleep()

    def extract_objects(self, results: ultralytics.engine.results.Results):
        objects = []
        for box in results.boxes:
            if self.yolo_model.names[int(box.cls)] == "person":
                continue
            else:
                objects.append(
                    {
                        "bbox": box.xyxy[0].cpu().numpy(),
                        "cls": box.cls.cpu().numpy(),
                        "conf": box.conf.item(),
                    }
                )
        objects = sorted(objects, key=lambda o: o["conf"], reverse=True)

        object_array = np.full((self.MAX_NR_OF_OBJECTS, 6), np.nan, dtype=np.float32)
        for ii in range(len(objects)):
            try:
                object_array[ii, :4] = objects[ii]["bbox"]
                object_array[ii, 4] = objects[ii]["cls"]
                object_array[ii, 5] = objects[ii]["conf"]
            except IndexError:
                break

        return object_array


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = YOLOLabeler(participant)
    node.run()
