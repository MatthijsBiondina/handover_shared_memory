import time

from cyclonedds.domain import DomainParticipant


class CycloneParticipant(DomainParticipant):
    RATE_HZ = 15

    def __init__(self, *args, **kwargs):
        super(CycloneParticipant, self).__init__()

        self.__T = time.time()

    def sleep(self):
        dt = time.time() - self.__T
        if dt < 1 / self.RATE_HZ:
            time.sleep(1 / self.RATE_HZ - dt)
        self.__T = time.time()
