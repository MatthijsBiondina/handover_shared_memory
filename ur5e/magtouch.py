import time
import numpy as np
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.defaults.float_idl import FloatSample
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.ddswriter import DDSWriter
from sensor_comm_dds.communication.data_classes.magtouch4 import MagTouch4

logger = get_logger()


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.mag = DDSReader(
            participant,
            topic_name=CYCLONE_NAMESPACE.MAGTOUCH_RAW,
            idl_dataclass=MagTouch4,
        )


class Writers:
    def __init__(self, participant: CycloneParticipant):
        self.mag_processed = DDSWriter(
            participant,
            topic_name=CYCLONE_NAMESPACE.MAGTOUCH_PROCESSED,
            idl_dataclass=FloatSample,
        )


class MagTouchProcessor:
    def __init__(self, participant: CycloneParticipant, N=30):
        self.N = N
        self.participant = participant
        self.readers = Readers(participant)
        self.writers = Writers(participant)

        self.timestamps = np.full((self.N,), np.nan)
        self.buffer = np.full((12, self.N), np.nan)
        self.mu, self.std = self.calibrate()

        logger.info("MagTouchProcessor: Ready!")

    def calibrate(self):
        while np.any(np.isnan(self.buffer)):
            try:
                mag = self.get_measurement()
                self.update_buffer(mag)
            except ContinueException:
                pass
            finally:
                self.participant.sleep()
        mu = np.mean(self.buffer, axis=-1)
        std = np.std(self.buffer, axis=-1)
        return mu, std

    def run(self):
        while True:
            try:
                mag = self.get_measurement()
                self.update_buffer(mag)
                diff = self.compute_slope()

                maxdiff = np.max(np.absolute(diff))

                msg = FloatSample(timestamp=time.time(), value=maxdiff)
                self.writers.mag_processed(msg)

            except ContinueException:
                pass
            finally:
                self.participant.sleep()

    def get_measurement(self) -> MagTouch4:
        mag: MagTouch4 = self.readers.mag.take()
        if mag is None:
            raise ContinueException
        return mag

    def update_buffer(self, m: MagTouch4):
        readings = np.array([val for txl in m.taxels for val in [txl.x, txl.y, txl.z]])
        self.buffer = np.roll(self.buffer, -1, axis=-1)
        self.buffer[:, -1] = readings
        self.timestamps = np.roll(self.timestamps, -1, axis=-1)
        self.timestamps[-1] = time.time()

    def compute_slope(self):
        x = self.timestamps[None, :]
        y = (self.buffer - self.mu[:, None]) / self.std[:, None]  # (bs, N)

        # Compute means for x and y
        x_mean = np.mean(x, axis=1, keepdims=True)
        y_mean = np.mean(y, axis=1, keepdims=True)

        numerator = np.sum((x - x_mean) * (y - y_mean), axis=1)
        denominator = np.sum((x - x_mean) ** 2, axis=1)

        slopes = numerator / denominator

        return slopes


if __name__ == "__main__":
    participant = CycloneParticipant(rate_hz=60)
    node = MagTouchProcessor(participant)
    node.run()
