import time

import numpy as np

from cantrips.debugging.terminal import pyout
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.curobo.collision_spheres_sample import CuroboCollisionSpheresSample
from cyclone.idl.procedures.coordinate_sample import CoordinateSample
from cyclone.idl.ur5e.tcp_pose_sample import TCPPoseSample
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
        self.grasp = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.GRASP_TCP_POSE,
            idl_dataclass=TCPPoseSample,
        )


class ApproachObjectProcedure:
    START_JS = [180, -180, 90, 90, 90, 0]
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

        grasp_timestamp, grasp_tcp = None, None

        while True:
            try:
                if not grasp_timestamp is None:
                    if time.time() - grasp_timestamp > self.PATIENCE:
                        grasp_timestamp = None
                        grasp_tcp = None

                if grasp_tcp is not None:
                    self.ur5e.move_to_tcp_pose(grasp_tcp)
                    raise ContinueException

                tgt: CoordinateSample = self.readers.target()
                if tgt is None or tgt.timestamp < time.time() - self.PATIENCE:
                    self.ur5e.move_to_tcp_pose(target_pose=tcp_rest)
                    raise ContinueException
                if not (0 > tgt.x > -1 and -0.5 < tgt.y < 0.5 and tgt.z > 0.2):
                    self.ur5e.move_to_tcp_pose(target_pose=tcp_rest)
                    raise ContinueException

                T = np.array([tgt.x, tgt.y, tgt.z])
                X = T.copy()
                X[:2] = T[:2] - 0.2 * (T[:2] / np.linalg.norm(T[:2]))
                tcp = self.ur5e.look_at(X, T)
                if self.ur5e.is_at_tcp_pose(tcp, pos_tol=0.02):
                    grasp: TCPPoseSample = self.readers.grasp()
                    if grasp is not None and time.time() - grasp.timestamp < 5:
                        grasp_timestamp = time.time()
                        grasp_tcp = np.array(grasp.pose)

                self.ur5e.move_to_tcp_pose(target_pose=tcp)

                # grasp: TCPPoseSample = self.readers.grasp()
                # if grasp is None or grasp.timestamp < time.time() - 10:
                #     self.ur5e.move_to_tcp_pose(target_pose=tcp_rest)
                #     raise ContinueException

                # grasp_tcp = np.array(grasp.pose)
                # x = grasp_tcp[0, 3]
                # y = grasp_tcp[1, 3]
                # z = grasp_tcp[2, 3]
                # if -1 < x < 0 and -0.5 < y < 0.5 and z > 0.2:
                #     self.ur5e.move_to_tcp_pose(target_pose=grasp_tcp)
                # else:
                #     self.ur5e.move_to_tcp_pose(target_pose=tcp_rest)

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
