import cv2
import numpy as np

from cantrips.configs import load_config
from cantrips.debugging.terminal import pyout, hex2rgb, UGENT
from cantrips.exceptions import ContinueException
from computer_vision.pointclouds import PointClouds
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.procedures.coordinate_sample import CoordinateSample
from cyclone.idl_shared_memory.points_idl import PointsIDL
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.sm_reader import SMReader
from visualization.webimagestreamer import WebImageStreamer


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.target = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.TARGET_OBJECT,
            idl_dataclass=CoordinateSample,
        )
        self.points = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_POINTCLOUD,
            idl_dataclass=PointsIDL(),
        )


class TargetPlotter:
    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()
        self.participant = participant
        self.readers = Readers(participant)
        self.web_streamer = WebImageStreamer(
            title="Focus", port=5005
        )
        

    def run(self):
        while True:
            try:
                points: PointsIDL = self.readers.points()
                target: CoordinateSample = self.readers.target()
                if points is None or target is None:
                    raise ContinueException

                obj_xyz = np.array([[target.x, target.y, target.z]])
                obj_uv = PointClouds.xyz2uv(obj_xyz, points.intrinsics, points.extrinsics)

                u, v = obj_uv[0, 0], obj_uv[0, 1]
                img = cv2.circle(
                    points.color,
                    (u, v), radius=5, color=hex2rgb(UGENT.BLUE), thickness=-1
                )
                self.web_streamer.update_frame(img[..., ::-1])
            except ContinueException:
                pass
            finally:
                self.participant.sleep()


if __name__ == "__main__":
    participant: CycloneParticipant = CycloneParticipant()
    node = TargetPlotter(participant)
    node.run()
