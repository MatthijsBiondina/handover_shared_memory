import time

import numpy as np

from cantrips.debugging.terminal import pyout
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.curobo.collision_spheres_sample import CuroboCollisionSpheresSample
from cyclone.idl.procedures.coordinate_sample import CoordinateSample
from cyclone.patterns.ddsreader import DDSReader
from ur5e.ur5e_client import Ur5eClient

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


class ApproachObjectProcedure:
    START_JS = [170, -100, 30, 90, 90, 0]
    PATIENCE = 10

    def __init__(self, domain_participant: CycloneParticipant):
        self.domain_participant = domain_participant
        self.ur5e = Ur5eClient(self.domain_participant)
        self.readers = Readers(self.domain_participant)

        logger.info("Approach object: Ready!")

    def run(self):
        self.wait_for_planner()
        self.ur5e.move_to_joint_configuration(np.deg2rad(self.START_JS), wait=True)
        tcp_rest = self.ur5e.tcp_pose

        tcp_look = np.array(
            [
                [0.0, 0.0, -1.0, -0.75],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        while True:
            try:
                self.ur5e.move_to_tcp_pose(tcp_look)
                # target = self.readers.target()
                # if (
                #     target is None
                #     or time.time() > target.timestamp + self.PATIENCE
                #     or target.z < 0.5
                # ):
                #     self.ur5e.move_to_tcp_pose(target_pose=tcp_rest)
                # else:
                #     object_pose = np.array([target.x, target.y, target.z])
                #     gripper_pose = object_pose - np.array([-0.2, 0.0, 0.0])
                #     tcp = self.ur5e.look_at(gripper_pose, object_pose)
                #     self.ur5e.move_to_tcp_pose(target_pose=tcp)

            except ContinueException:
                pass
            finally:
                self.domain_participant.sleep()

        # pyout()

    def move_to_approach_pose(self):
        middle = np.array([-0.8, 0.0, 0.5])
        focus = np.array([-1.5, 0.15, 0.35])
        tcp = self.ur5e.look_at(middle, focus)
        self.ur5e.move_to_tcp_pose(tcp, wait=True)

        def pose(t):
            progress = t / 10
            theta = 2 * np.pi * progress
            dy = 0.3 * np.cos(theta)
            tcp = self.ur5e.look_at(middle + np.array([0, dy, 0]), focus)
            return tcp

        t0 = time.time()
        while time.time() < t0 + 30000:
            self.ur5e.move_to_tcp_pose(pose(time.time()))
            self.domain_participant.sleep()

    def wait_for_planner(self):
        while True:
            sphere = self.readers.spheres()
            if sphere is not None:
                break
            self.domain_participant.sleep()


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = ApproachObjectProcedure(participant)
    node.run()
