import time

import numpy as np

from cantrips.debugging.terminal import pyout
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.procedures.coordinate_sample import CoordinateSample
from cyclone.idl.sensor_fusion.kalman_sample import KalmanSample
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.ddswriter import DDSWriter

logger = get_logger()

class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.objects = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.KALMAN_OBJECTS,
            idl_dataclass=KalmanSample,
        )
        self.hands = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.KALMAN_HANDS,
            idl_dataclass=KalmanSample,
        )


class Writers:
    def __init__(self, participant: CycloneParticipant):
        self.target = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.TARGET_OBJECT,
            idl_dataclass=CoordinateSample,
        )


class TargetObjectChooser:
    HOLDING_DISTANCE_THRESHOLD = 0.25

    def __init__(self, participant: CycloneParticipant):
        self.participant = participant
        self.readers = Readers(participant)
        self.writers = Writers(participant)
        logger.info("TargetObjectChooser: Ready!")

    def run(self):
        while True:
            try:
                hands_sample: KalmanSample = self.readers.hands()
                obj_sample: KalmanSample = self.readers.objects()
                if hands_sample is None or obj_sample is None:
                    raise ContinueException

                hands = self.preprocess(hands_sample)
                objects = self.preprocess(obj_sample)
                if hands.size == 0 or objects.size == 0:
                    raise ContinueException

                _, object = self.pair_objects_with_hands(hands, objects)

                msg = CoordinateSample(
                    timestamp=time.time(), x=object[0], y=object[1], z=object[2]
                )
                self.writers.target(msg)
            except ContinueException:
                pass
            finally:
                self.participant.sleep()

    def preprocess(self, sample: KalmanSample) -> np.ndarray:
        xyz = np.array(sample.mean)
        if xyz.size:
            mask = ~np.any(np.isnan(xyz), axis=1)
            return xyz[mask]
        else:
            raise ContinueException

    def pair_objects_with_hands(self, hands: np.ndarray, objects: np.ndarray):
        D = np.linalg.norm(hands[:, None, :] - objects[None, :, :], axis=-1)

        closest_idx = np.argmin(D, axis=1)

        # For each hand, get the object that is closest to it and within threshold distance
        pairs = []
        for hand_idx, obj_idx in enumerate(closest_idx):
            if D[hand_idx, obj_idx] < self.HOLDING_DISTANCE_THRESHOLD:
                pairs.append((hands[hand_idx], objects[obj_idx]))

        if len(pairs) == 0:
            raise ContinueException

        # If more than one pair, get the hand closest to the robot
        closest_pair = sorted(pairs, key=lambda p: np.linalg.norm(p[0]))[0]

        return closest_pair


if __name__ == "__main__":
    participant: CycloneParticipant = CycloneParticipant()
    node = TargetObjectChooser(participant)
    node.run()
