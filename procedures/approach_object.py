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
    START_JS = [180, -180, 90, 90, 90, 0]
    # START_JS = [115, -180, 90, 90, 90, 0]

    WORKSPACE = {
        "xmin": -1,
        "xmax": 0,
        "ymin": -0.5,
        "ymax": 0.5,
        "zmin": 0.2,
        "zmax": 1.5,
    }
    PATIENCE = 20
    GRASP_OPTIMIZATION_TIME = 5
    GIVE_BACK_TCP = np.array(
        [
            [0.0, 0.0, -1.0, -0.75],
            [-1.0, 0.0, 0.0, -0.2],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    def __init__(self, domain_participant: CycloneParticipant):
        self.domain_participant = domain_participant
        self.ur5e = Ur5eClient(self.domain_participant)
        self.readers = Readers(self.domain_participant)
        self.writers = Writers(self.domain_participant)

        self.tcp_rest, self.state = self.initialize()

        self.stopwatch = None
        self.grasp_tcp = None
        self.give_back_tcp = self.GIVE_BACK_TCP
        self.grasping = False

        logger.info("Approach object: Ready!")

    def initialize(self):
        while True:
            sphere = self.readers.spheres()
            if sphere is not None:
                break
            self.domain_participant.sleep()
        self.ur5e.move_to_joint_configuration(np.deg2rad(self.START_JS), wait=True)
        self.tcp_rest = self.ur5e.tcp_pose
        if self.ur5e.is_holding_an_object:
            self.state = States.GIVE_BACK
        else:
            self.state = States.RESTING
            self.ur5e.open_gripper()
        return self.tcp_rest, self.state

    def run(self):
        state = "INIT"
        while True:
            try:
                self.writers.state(StateSample(time.time(), self.state))
                if self.state != state:
                    state = self.state
                    logger.info(States(self.state))

                if self.state == States.RESTING:
                    self.state = self.rest()
                elif self.state == States.APPROACHING:
                    self.state = self.approach()
                elif self.state == States.REACHING:
                    self.state = self.reach()
                    if not self.state == States.REACHING:
                        self.stopwatch = None
                        self.grasp_tcp = None
                elif self.state == States.GRASPING:
                    self.state = self.grasp()
                elif self.state == States.RETRACT:
                    self.state = self.retract()
                elif self.state == States.GIVE_BACK:
                    self.state = self.give_back()
            except ContinueException:
                pass
            finally:
                self.domain_participant.sleep()

    def rest(self):
        self.ur5e.open_gripper()
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

        if self.ur5e.is_at_tcp_pose(tcp, pos_tol=0.25, rot_tol=None):
            return States.REACHING

        return States.APPROACHING

    def reach(self):
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
                    if dt > self.GRASP_OPTIMIZATION_TIME + self.PATIENCE:
                        return States.RESTING
                    else:
                        raise ContinueException
                self.grasp_tcp = np.array(grasp.pose)
                self.give_back_tcp = np.array(grasp.pose)
            self.ur5e.move_to_tcp_pose(self.grasp_tcp)
            if self.ur5e.is_at_tcp_pose(self.grasp_tcp, pos_tol=0.01, rot_tol=None):
                return States.GRASPING
            if dt > self.GRASP_OPTIMIZATION_TIME + 5:
                return States.GRASPING

        if dt > self.PATIENCE:
            return States.RESTING
        return States.REACHING

    def grasp(self):
        self.ur5e.close_gripper()
        if self.ur5e.is_holding_an_object:
            return States.RETRACT
        if self.ur5e.gripper_width < 0.01:
            return States.RESTING

        return States.GRASPING

    def retract(self):
        self.ur5e.move_to_tcp_pose(self.tcp_rest)
        if self.ur5e.is_at_tcp_pose(self.tcp_rest):
            if self.ur5e.is_holding_an_object:
                self.stopwatch = time.time()
                return States.GIVE_BACK
            else:
                self.ur5e.open_gripper()
                return States.RESTING
        return States.RETRACT

    def give_back(self):
        if self.stopwatch is None:
            self.stopwatch = time.time()

        self.ur5e.move_to_tcp_pose(self.give_back_tcp)
        if (
            self.ur5e.is_at_tcp_pose(self.give_back_tcp, rot_tol=5)
            or time.time() - self.stopwatch > 10
        ):
            self.ur5e.open_gripper()
            return States.RESTING
        return States.GIVE_BACK

    def is_target_in_workspace(self, target, zmin=0.2, cone=45, max_dist=1.0):
        if target is None:
            return False
        if time.time() - target.timestamp > self.PATIENCE:
            return False

        if target.z < zmin:
            return False
        tgt = np.array([target.x, target.y, target.z, 1.0])
        tgt_tcp = (np.linalg.inv(self.tcp_rest) @ tgt[:, None]).squeeze(-1)[:3]
        angle = np.rad2deg(np.arctan2(tgt_tcp[0], tgt_tcp[2]))
        if abs(angle) > cone:
            return False
        if np.linalg.norm(tgt[:2]) > max_dist:
            return False

        return True


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = ApproachObjectProcedure(participant)
    node.run()
