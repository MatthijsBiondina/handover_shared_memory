import time
from statistics import covariance
from typing import List

import numpy as np
import torch

from cantrips.exceptions import ContinueException, BreakException
from cantrips.logging.logger import get_logger
from computer_vision.pointclouds import PointClouds
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.sensor_fusion.kalman_sample import KalmanSample
from cyclone.idl_shared_memory.yolo_idl import YOLOIDL
from cyclone.patterns.ddswriter import DDSWriter
from cyclone.patterns.sm_reader import SMReader


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.yolo = SMReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.YOLO,
            idl_dataclass=YOLOIDL(),
        )


class Writers:
    def __init__(self, participant: CycloneParticipant):
        self.kalman = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.KALMAN_OBJECTS,
            idl_dataclass=KalmanSample,
        )

logger = get_logger()

class KalmanYOLO:
    CONFIDENCE_THRESHOLD = 0.2
    MEASUREMENT_ERROR_STD = 0.05
    MAHALANOBIS_THRESHOLD = 1
    MOTION_NOISE_STD = 0.1
    MAX_UNCERTAINTY = 1.0

    def __init__(self, participant: CycloneParticipant):
        self.participant = participant
        self.readers = Readers(participant)
        self.writers = Writers(participant)
        self.timestamp = 0

        self.mean: List[np.ndarray] = []
        self.covariance: List[np.ndarray] = []

        logger.info(f"YoloSensorFusion: Ready!")

    def run(self):
        while True:
            try:
                yolo_sample: YOLOIDL = self.readers.yolo()
                if yolo_sample.timestamp <= self.timestamp:
                    raise ContinueException
                else:
                    dt = yolo_sample.timestamp - self.timestamp
                    self.timestamp = yolo_sample.timestamp

                y = self.preprocess_measurement(yolo_sample)

                self.motion_update(dt)
                self.add_new_measurements(y)
                self.landmark_association()

                msg = KalmanSample(
                    timestamp=yolo_sample.timestamp,
                    mean=[mu.tolist() for mu in self.mean],
                    covariance=[Sigma.tolist() for Sigma in self.covariance],
                )
                self.writers.kalman(msg)
            except ContinueException:
                pass
            finally:
                self.participant.sleep()

    def preprocess_measurement(self, yolo_sample: YOLOIDL):
        objects = yolo_sample.objects
        measurements = objects[~np.any(np.isnan(objects), axis=1)]
        measurements = measurements[measurements[:, -1] > self.CONFIDENCE_THRESHOLD]

        u = (measurements[:, 0] + measurements[:, 2]) / 2
        v = (measurements[:, 1] + measurements[:, 3]) / 2

        uv = np.stack((u, v), axis=1)
        if uv.shape[0] == 0:
            return np.empty((0, 3))

        xyz = PointClouds.uv2xyz(uv, yolo_sample.depth, yolo_sample.points)
        return xyz

    def motion_update(self, dt: float):
        R = np.eye(3) * (self.MOTION_NOISE_STD * dt) ** 2
        for idx in range(len(self.mean)):
            self.covariance[idx] += R
        for idx in range(len(self.mean)-1, -1, -1):
            if np.any(np.diag(self.covariance[idx]) < self.MAX_UNCERTAINTY**2):
                del self.covariance[idx]
                del self.mean[idx]

    def add_new_measurements(self, xyz: np.ndarray):
        for mu in xyz:
            self.mean.append(mu)
            self.covariance.append(np.eye(3) * self.MEASUREMENT_ERROR_STD**2)

    def landmark_association(self):
        modified_flag = True
        while modified_flag:
            try:
                for idxA in range(0, len(self.mean)):
                    for idxB in range(0, len(self.mean)):
                        if idxA == idxB:
                            continue
                        association = self.associate_keypoints(idxA, idxB)
                        if association:
                            mu, Sigma = association
                            self.mean[idxA] = mu
                            self.covariance[idxA] = Sigma
                            del self.mean[idxB]
                            del self.covariance[idxB]
                            raise BreakException
            except BreakException:
                modified_flag = True
            else:
                modified_flag = False

    def associate_keypoints(self, idxA: int, idxB: int):
        # Equality measurement
        mu = np.concatenate((self.mean[idxA], self.mean[idxB]), axis=0)[:, None]
        Sigma = np.eye(6)
        Sigma[:3, :3] = self.covariance[idxA]
        Sigma[3:, 3:] = self.covariance[idxB]
        C = np.zeros((3, 6))
        C[:, :3] = np.eye(3)
        C[:, 3:] = -np.eye(3)
        y = np.zeros((3, 1))
        y_star = C @ mu
        Q = np.eye(3) * 1e-3

        # Kalman update
        K = Sigma @ C.T @ np.linalg.inv(C @ Sigma @ C.T + Q)
        mu_ = mu + K @ (y - y_star)
        Sigma_ = (np.eye(6) - K @ C) @ Sigma

        # Mahalanobis distance
        md = float(np.sqrt((mu_ - mu).T @ np.linalg.inv(Sigma) @ (mu_ - mu)).squeeze())

        if md < self.MAHALANOBIS_THRESHOLD:
            return mu_[:3, 0], Sigma_[:3, :3]
        else:
            return False


if __name__ == "__main__":
    with torch.no_grad():
        participant = CycloneParticipant()
        node = KalmanYOLO(participant)
        node.run()
