import json
import logging
import os
from typing import Dict, List

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
from cyclone.idl_shared_memory.zed_points_idl import ZedPointsIDL
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
            topic_name=CYCLONE_NAMESPACE.ZED_POINTCLOUD,
            idl_dataclass=ZedPointsIDL(),
        )


class Writers:
    def __init__(self, participant: CycloneParticipant):
        self.things = SMWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.YOLO_ZED,
            idl_dataclass=YOLOIDL(),
        )
        self.person = SMWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.YOLO_PERSON,
            idl_dataclass=YOLOIDL(),
        )


logger = get_logger()


class ZEDYOLOLabeler:
    def __init__(self, participant: CycloneParticipant):
        self.participant = participant
        self.readers = Readers(participant)
        self.writers = Writers(participant)
        self.MAX_NR_OF_OBJECTS = YOLOIDL().objects.shape[0]

        self.yolo_model = YOLO("yolov8x")

        with open(f"{os.path.dirname(__file__)}/../config/yolo_names.json", "w+") as f:
            json.dump(self.yolo_model.names, f, indent=2)

        logger.info("ZedYOLOLabeler: Ready!")

    def run(self):
        while True:
            try:
                frame: ZedPointsIDL = self.readers.frame()
                if frame is None:
                    raise ContinueException
                h, w, _ = frame.color.shape
                results_left = self.yolo_model(frame.color[:, :h])
                results_right = self.yolo_model(frame.color[:, -h:])

                objects, persons = [], []
                if len(results_left):
                    objects_left, persons_left = self.extract_objects(results_left[0])
                    objects.extend(objects_left)
                    persons.extend(persons_left)
                if len(results_right):
                    objects_right, persons_right = self.extract_objects(
                        results_right[0]
                    )
                    for o in objects_right:
                        o["bbox"] += np.array([w - h, 0.0, w - h, 0.0])
                    for p in persons_right:
                        p["bbox"] += np.array([w - h, 0.0, w - h, 0.0])
                    objects.extend(objects_right)
                    persons.extend(persons_right)

                self.writers.things(self.process_objects(objects, frame))
                self.writers.person(self.process_objects(persons, frame))

            except ContinueException:
                pass
            finally:
                self.participant.sleep()

    def extract_objects(self, results: ultralytics.engine.results.Results):
        objects, persons = [], []
        for box in results.boxes:
            measure = {
                "bbox": box.xyxy[0].cpu().numpy(),
                "cls": box.cls.cpu().numpy(),
                "conf": box.conf.item(),
            }
            if self.yolo_model.names[int(box.cls)] == "person":
                persons.append(measure)
            else:
                objects.append(measure)
        return objects, persons

    def process_objects(
        self, objects: List[Dict[str, np.ndarray | float]], frame: ZedPointsIDL
    ) -> YOLOIDL:
        objects = sorted(objects, key=lambda o: o["conf"], reverse=True)
        N = min(len(objects), self.MAX_NR_OF_OBJECTS)
        objects_array = np.full((N, 6), np.nan, dtype=np.float32)
        for ii in range(len(objects)):
            try:
                objects_array[ii, :4] = objects[ii]["bbox"]
                objects_array[ii, 4] = objects[ii]["cls"].item()
                objects_array[ii, 5] = objects[ii]["conf"]
            except IndexError:
                break
        objects_uv = np.stack(
            (
                (objects_array[:, 0] + objects_array[:, 2]) / 2,
                (objects_array[:, 1] + objects_array[:, 3]) / 2,
            ),
            axis=-1,
        ).astype(int)
        objects_xyz = frame.points[objects_uv[:, 1], objects_uv[:, 0]]
        msg = YOLOIDL(
            timestamp=frame.timestamp,
            extrinsics=frame.extrinsics,
            intrinsics=frame.intrinsics,
            objects=self.pad_array(objects_array),
            uv=self.pad_array(objects_uv),
            xyz=self.pad_array(objects_xyz),
        )
        return msg

    def pad_array(self, arr: np.ndarray):
        arr_padded = np.full(
            (self.MAX_NR_OF_OBJECTS, arr.shape[1]), np.nan, dtype=arr.dtype
        )
        arr_padded[: arr.shape[0]] = arr
        return arr_padded


if __name__ == "__main__":
    participant = CycloneParticipant(rate_hz=5)
    node = ZEDYOLOLabeler(participant)
    node.run()
