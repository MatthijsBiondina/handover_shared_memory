import time
from cyclonedds.domain import DomainParticipant


class CycloneParticipant(DomainParticipant):
    RATE_HZ = 15

    def __init__(self, rate_hz=None):
        super(CycloneParticipant, self).__init__()
        # Make sure to use a number, not the class itself
        if isinstance(rate_hz, int):
            self._sleep_interval = rate_hz
        else:
            self._sleep_interval = self.RATE_HZ
        self.__T = time.time()

    def sleep(self):
        dt = time.time() - self.__T
        if dt < 1 / self._sleep_interval:  # Now this will use the numeric value
            time.sleep(1 / self._sleep_interval - dt)
        self.__T = time.time()
