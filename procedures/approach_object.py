import time

import numpy as np

from cantrips.debugging.terminal import pyout
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.curobo.collision_spheres_sample import CuroboCollisionSpheresSample
from cyclone.patterns.ddsreader import DDSReader
from ur5e.ur5e_client import Ur5eClient


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.spheres = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.CUROBO_COLLISION_SPHERES,
            idl_dataclass=CuroboCollisionSpheresSample,
        )


class ApproachObjectProcedure:
    START_JS = [170, -100, 30, 90, 90, 0]

    def __init__(self, domain_participant: CycloneParticipant):
        self.domain_participant = domain_participant
        self.ur5e = Ur5eClient(self.domain_participant)
        self.readers = Readers(self.domain_participant)

    def run(self):
        self.wait_for_planner()
        self.ur5e.move_to_joint_configuration(np.deg2rad(self.START_JS), wait=True)
        time.sleep(2)
        self.circle()
        self.ur5e.move_to_joint_configuration(np.deg2rad(self.START_JS), wait=True)


        # pyout()

    def circle(self):
        middle = np.array([-0.8, 0., 0.5])
        focus = np.array([-1.5, 0.15, 0.35])
        tcp = self.ur5e.look_at(middle, focus)
        self.ur5e.move_to_tcp_pose(tcp, wait=True)

        def pose(t):
            progress = t / 10
            theta = 2 * np.pi * progress
            dy = 0.3*np.cos(theta)
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
