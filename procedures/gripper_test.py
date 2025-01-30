import sys
import time

import numpy as np

from cantrips.debugging.terminal import pyout
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.curobo.collision_spheres_sample import CuroboCollisionSpheresSample
from cyclone.idl.defaults.boolean_idl import StateSample
from cyclone.idl.procedures.coordinate_sample import CoordinateSample
from cyclone.idl.ur5e.tcp_pose_sample import TCPPoseSample
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.ddswriter import DDSWriter
from procedures.state_machine_states import States
from ur5e.ur5e_client import Ur5eClient
from ur5e.ur5e_utils import compute_approach_pose

logger = get_logger()


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.spheres = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.CUROBO_COLLISION_SPHERES,
            idl_dataclass=CuroboCollisionSpheresSample,
        )

class Writers:
    def __init__(self, participant: CycloneParticipant):
        pass

class GripperTestProcedure:
    JOINTS= [190, -4, 0, 4, 100, 0]

    def __init__(self, domain_participant = CycloneParticipant):
        self.participant = domain_participant
        self.ur5e = Ur5eClient(self.participant)
        self.readers = Readers(self.participant)
        self.writers = Writers(self.participant)
        self.tcp_rest = self.initialize()
        logger.info("GripperTestProcedure: Ready!")

    def initialize(self):
        while True:
            sphere = self.readers.spheres()
            if sphere is not None:
                break
            self.participant.sleep()
        self.ur5e.move_to_joint_configuration(np.deg2rad(self.JOINTS), wait=True)
        self.tcp_rest = self.ur5e.tcp_pose
        return self.tcp_rest

    def run(self, interval=10):
        while True:
            # self.ur5e.open_gripper()
            # sys.exit(0)
            try:
                self.ur5e.close_gripper()
                time.sleep(interval)
                self.ur5e.open_gripper()
                time.sleep(interval)
            except ContinueException:
                pass
            finally:
                self.participant.sleep()

if __name__ == "__main__":
    participant = CycloneParticipant()
    node = GripperTestProcedure(participant)
    node.run()
        