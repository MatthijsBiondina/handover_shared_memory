import time

import numpy as np

from cantrips.debugging.terminal import pyout
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.curobo.collision_spheres_sample import CuroboCollisionSpheresSample
from cyclone.patterns.ddsreader import DDSReader
from ur5e.ur5e_client import Ur5eClient

np.set_printoptions(precision=2, suppress=True)

logger = get_logger()

class Readers:
    def __init__(self, paticipant: CycloneParticipant):
        self.spheres = DDSReader(
            domain_participant=paticipant,
            topic_name=CYCLONE_NAMESPACE.CUROBO_COLLISION_SPHERES,
            idl_dataclass=CuroboCollisionSpheresSample,
        )

class LookAtDeskProcedure:
    START_JS = [170, -140, 100, 40, 90, 0]

    def __init__(self, domain_participant: CycloneParticipant):
        self.participant = domain_participant
        self.ur5e = Ur5eClient(self.participant)
        self.readers = Readers(self.participant)

    def run(self):
        while True:
            sphere = self.readers.spheres()
            if sphere is not None:
                break
            self.participant.sleep()

        self.ur5e.move_to_joint_configuration(np.deg2rad(self.START_JS), wait=True)
        self.circles()


        logger.warning(f"{np.rad2deg(self.ur5e.joint_state)}")

    def circles(self, duration=None):
        Y, Z, r, T = 0.0, 0.45, 0.001, 10

        def pose(t):
            progress = t / T
            theta = 2 * np.pi * progress
            y = Y + r * np.cos(theta)
            z = Z + r * np.sin(theta)

            return np.array(
                [
                    [0.0, 0.0, -1.0, -0.6],
                    [-1.0, 0.0, 0.0, y+0.25],
                    [0.0, 1.0, 0.0, z],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

        t0 = time.time()
        while duration is None or time.time() - t0 < duration:
            t = time.time() - t0
            self.ur5e.move_to_tcp_pose(pose(t))
            self.participant.sleep()


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = LookAtDeskProcedure(participant)
    node.run()
