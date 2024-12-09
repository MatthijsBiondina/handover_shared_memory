import cv2
import numpy as np
import torch

from cantrips.configs import load_config
from cantrips.debugging.terminal import pyout, UGENT, hex2rgb
from cantrips.exceptions import ContinueException
from computer_vision.pointclouds import PointClouds
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.sensor_fusion.kalman_sample import KalmanSample
from cyclone.idl_shared_memory.points_idl import PointsIDL
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.sm_reader import SMReader
from visualization.webimagestreamer import WebImageStreamer


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.objects = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.KALMAN_OBJECTS,
            idl_dataclass=KalmanSample,
        )
        self.hands = DDSReader(
            domain_participant = participant,
            topic_name=CYCLONE_NAMESPACE.KALMAN_HANDS,
            idl_dataclass=KalmanSample,
        )
        self.points = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.D405_POINTCLOUD,
            idl_dataclass=PointsIDL(),
        )


class KalmanPlotter:
    UNCERTAINTY_THRESHOLD_STD = 0.2

    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()
        self.participant = participant
        self.readers = Readers(participant)
        self.web_streamer = WebImageStreamer(
            title="Long-Term Object Positions", port=5004
        )

    def run(self):
        while True:
            try:
                points: PointsIDL = self.readers.points()
                objects: KalmanSample = self.readers.objects()
                hands: KalmanSample = self.readers.hands()
                if points is None or objects is None or hands is None:
                    raise ContinueException
                xyz_obj = self.extract_objects(objects)
                xyz_hands = self.extract_objects(hands)

                img = points.color
                if xyz_obj.shape[0] > 0:
                    uv_obj = PointClouds.xyz2uv(xyz_obj, points.intrinsics, points.extrinsics)
                    img = self.draw_objects(img, uv_obj)
                if xyz_hands.shape[0] > 0:
                    uv_hands = PointClouds.xyz2uv(xyz_hands, points.intrinsics, points.extrinsics)
                    img = self.draw_objects(img, uv_hands, color=UGENT.GREEN)
                self.web_streamer.update_frame(img[..., ::-1])
            except ContinueException:
                pass
            finally:
                self.participant.sleep()

    def extract_objects(self, objects: KalmanSample):
        xyz = []
        for mu, Sigma in zip(objects.mean, objects.covariance):
            if np.all(np.diag(np.array(Sigma)) < self.UNCERTAINTY_THRESHOLD_STD**2):
                xyz.append(mu)
        return np.array(xyz)

    def draw_objects(self, img: np.ndarray, objects: np.ndarray, color=UGENT.BLUE):
        for u, v in objects:
            img = cv2.circle(
                img, (u, v), radius=5, color=hex2rgb(color), thickness=-1
            )
        return img


if __name__ == "__main__":
    with torch.no_grad():
        participant = CycloneParticipant()
        node = KalmanPlotter(participant)
        node.run()
