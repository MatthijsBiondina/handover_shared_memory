from typing import List

import numpy as np

from cantrips.debugging.terminal import pyout
from cantrips.exceptions import ContinueException, BreakException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.sensor_fusion.kalman_sample import KalmanSample
from cyclone.idl_shared_memory.mediapipe_idl import MediapipeIDL
from cyclone.patterns.ddswriter import DDSWriter
from cyclone.patterns.sm_reader import SMReader

logger = get_logger()


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.hands = SMReader(
            participant,
            topic_name=CYCLONE_NAMESPACE.MEDIAPIPE_POSE,
            idl_dataclass=MediapipeIDL(),
        )


class Writers:
    def __init__(self, participant: CycloneParticipant):
        self.hand_centroids = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.KALMAN_HANDS,
            idl_dataclass=KalmanSample,
        )


class KalmanHands:
    MOTION_NOISE_STD = 0.5
    MAX_UNCERTAINTY = 1.0
    MEASUREMENT_ERROR_STD = 0.1
    MAHALANOBIS_THRESHOLD = 1

    def __init__(self, participant: CycloneParticipant):
        self.participant = participant
        self.readers = Readers(participant)
        self.writers = Writers(participant)
        self.timestamp = 0

        self.mean: List[np.ndarray] = []
        self.covariance: List[np.ndarray] = []

        logger.info(f"HandsSensorFusion: Ready!")

    def run(self):
        while True:
            try:
                sample: MediapipeIDL = self.readers.hands()
                if sample is None:
                    raise ContinueException
                if sample.timestamp <= self.timestamp:
                    raise ContinueException
                dt = sample.timestamp - self.timestamp
                self.timestamp = sample.timestamp
                y = self.preprocess_measurement(sample)

                self.motion_update(dt)
                self.add_new_measurements(y)
                self.landmark_association()

                msg = KalmanSample(
                    timestamp=sample.timestamp,
                    mean=[mu.tolist() for mu in self.mean],
                    covariance=[Sigma.tolist() for Sigma in self.covariance],
                )
                self.writers.hand_centroids(msg)
            except ContinueException:
                pass
            finally:
                self.participant.sleep()

    def preprocess_measurement(self, sample: MediapipeIDL):
        hands = sample.xyz
        mask = ~np.any(np.isnan(hands), axis=1)
        return hands[mask]

    def motion_update(self, dt: float):
        R = np.eye(3) * (self.MOTION_NOISE_STD * dt) ** 2
        for idx in range(len(self.mean)):
            self.covariance[idx] += R
        for idx in range(len(self.mean) - 1, -1, -1):
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
                            self.mean[idxA], self.covariance[idxA] = association
                            del self.mean[idxB], self.covariance[idxB]
                            raise BreakException
            except BreakException:
                modified_flag = True
            else:
                modified_flag = False

    def associate_keypoints(self, idxA: int, idxB: int):
        # Equaity measurement
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
    participant = CycloneParticipant()
    node = KalmanHands(participant)
    node.run()
