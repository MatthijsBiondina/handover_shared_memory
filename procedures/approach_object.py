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
        self.target = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.TARGET_OBJECT,
            idl_dataclass=CoordinateSample,
        )
        self.grasp = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.GRASP_TCP_POSE,
            idl_dataclass=TCPPoseSample,
        )


class Writers:
    def __init__(self, participant: CycloneParticipant):
        self.state = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.STATE_MACHINE_STATE,
            idl_dataclass=StateSample,
        )


class ApproachObjectProcedure:
    START_JS = [180, -100, 30, 90, 90, 0]
    WORKSPACE = {
        "xmin": -1,
        "xmax": 0,
        "ymin": -0.5,
        "ymax": 0.5,
        "zmin": 0.2,
        "zmax": 1.5,
    }
    PATIENCE = 20
    GRASP_OPTIMIZATION_TIME = 3

    def __init__(self, domain_participant: CycloneParticipant):
        self.domain_participant = domain_participant
        self.ur5e = Ur5eClient(self.domain_participant)
        self.readers = Readers(self.domain_participant)
        self.writers = Writers(self.domain_participant)

        self.state = States.RESTING
        self.tcp_rest = self.initialize()

        self.stopwatch = None
        self.grasp_tcp = None

        logger.info("Approach object: Ready!")

    def initialize(self):
        while True:
            sphere = self.readers.spheres()
            if sphere is not None:
                break
            self.domain_participant.sleep()
        self.ur5e.move_to_joint_configuration(np.deg2rad(self.START_JS), wait=True)
        self.tcp_rest = self.ur5e.tcp_pose
        return self.tcp_rest

    def run(self):
        while True:
            try:
                self.writers.state(StateSample(time.time(), self.state))
                if self.state == States.RESTING:
                    self.state = self.rest()
                if self.state == States.APPROACHING:
                    self.state = self.approach()
                if self.state == States.GRASPING:
                    self.state = self.grasp()
                    if not self.state == States.GRASPING:
                        self.stopwatch = None
                        self.grasp_tcp = None
            except ContinueException:
                pass
            finally:
                self.domain_participant.sleep()

    def rest(self):
        self.ur5e.move_to_tcp_pose(self.tcp_rest)

        # Check whether target object spotted
        target = self.readers.target()
        if self.is_target_in_workspace(target):
            return States.APPROACHING
        return States.RESTING

    def approach(self):
        target = self.readers.target()
        if not self.is_target_in_workspace(target):
            return States.RESTING

        tcp = compute_approach_pose(target.x, target.y, target.z)
        self.ur5e.move_to_tcp_pose(tcp)

        if self.ur5e.is_at_tcp_pose(tcp, pos_tol=0.02, rot_tol=20):
            return States.GRASPING

        return States.APPROACHING

    def grasp(self):
        if self.stopwatch is None:
            self.stopwatch = time.time()

        dt = time.time() - self.stopwatch
        if dt < self.GRASP_OPTIMIZATION_TIME:
            target = self.readers.target()
            if not self.is_target_in_workspace(target):
                return States.RESTING
            else:
                tcp = compute_approach_pose(target.x, target.y, target.z)
                self.ur5e.move_to_tcp_pose(tcp)
        else:
            if self.grasp_tcp is None:
                grasp = self.readers.grasp()
                if grasp is None:
                    raise ContinueException
                self.grasp_tcp = np.array(grasp.pose)
            self.ur5e.move_to_tcp_pose(self.grasp_tcp)

        if dt > self.PATIENCE:
            return States.RESTING
        return States.GRASPING

    def is_target_in_workspace(self, target):
        if target is None:
            return False
        if time.time() - target.timestamp > self.PATIENCE:
            return False
        if not (
            self.WORKSPACE["xmin"] < target.x < self.WORKSPACE["xmax"]
            and self.WORKSPACE["ymin"] < target.y < self.WORKSPACE["ymax"]
            and self.WORKSPACE["zmin"] < target.z < self.WORKSPACE["zmax"]
        ):
            return False
        return True


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = ApproachObjectProcedure(participant)
    node.run()
