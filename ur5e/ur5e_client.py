import time

import numpy as np
from cyclonedds.domain import DomainParticipant
from spatialmath.base import transl

from cantrips.debugging.terminal import pyout
from cantrips.exceptions import WaitingForFirstMessageException
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.ur5e.joint_configuration_sample import JointConfigurationSample
from cyclone.idl.ur5e.tcp_pose_sample import TCPPoseSample
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.ddswriter import DDSWriter


class Readers:
    def __init__(self, participant: DomainParticipant):
        self.joint_configuration = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.UR5E_JOINT_CONFIGURATION,
            idl_dataclass=JointConfigurationSample,
        )
        self.tcp_pose = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.UR5E_TCP_POSE,
            idl_dataclass=TCPPoseSample,
        )


class Writers:
    def __init__(self, participant: DomainParticipant):
        self.goal_js = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.TARGET_JOINT_CONFIGURATION,
            idl_dataclass=JointConfigurationSample,
        )
        self.goal_tcp = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.TARGET_TCP_POSE,
            idl_dataclass=TCPPoseSample,
        )


class Ur5eClient:
    def __init__(self, participant: CycloneParticipant):
        self.participant = participant
        self.readers = Readers(participant)
        self.writers = Writers(participant)
        self.idle(1.0)

    def idle(self, duration: float=0.5):
        t0 = time.time()
        while time.time() - t0 < duration:
            self.move_to_tcp_pose(self.tcp_pose)
            self.participant.sleep()

    @property
    def joint_state(self):
        while True:
            try:
                sample = self.readers.joint_configuration()
                if sample is None:
                    raise WaitingForFirstMessageException
                return np.array(sample.pose)
            except WaitingForFirstMessageException:
                self.participant.sleep()

    @property
    def tcp_pose(self):
        while True:
            try:
                sample = self.readers.tcp_pose()
                if sample is None:
                    raise WaitingForFirstMessageException
                return np.array(sample.pose)
            except WaitingForFirstMessageException:
                self.participant.sleep()

    def move_to_joint_configuration(
        self, target_joints: np.ndarray, wait: bool = True
    ) -> None:
        assert np.all(target_joints >= -2 * np.pi) and np.all(
            target_joints <= 2 * np.pi
        )
        self.idle()
        msg = JointConfigurationSample(
            timestamp=time.time(),
            pose=target_joints.tolist(),
            velocity=np.zeros_like(target_joints.tolist()),
        )
        self.writers.goal_js(msg)

        if wait:
            while not self.is_at_joint_state(target_joints):
                self.participant.sleep()
        self.idle()

    def move_to_tcp_pose(
        self, target_pose: np.ndarray, wait: bool = False, timeout=5.0
    ):
        msg = TCPPoseSample(
            timestamp=time.time(),
            pose=target_pose.tolist(),
            velocity=np.zeros_like(target_pose).tolist(),
        )
        self.writers.goal_tcp(msg)

        if wait:
            t0 = time.time()
            while not self.is_at_tcp_pose(target_pose) and time.time() - t0 < timeout:
                self.participant.sleep()

        return self.is_at_tcp_pose(target_pose)

    def is_at_joint_state(self, joints: np.ndarray):
        return np.all(np.isclose(self.joint_state, joints, atol=0.01))

    def is_at_tcp_pose(self, pose: np.ndarray):
        return np.all(np.isclose(self.tcp_pose, pose, atol=0.01))

    def look_at(self, position: np.ndarray, focus: np.ndarray):
        z_axis = focus - position
        z_axis = z_axis / np.linalg.norm(z_axis)

        x_axis = np.cross(np.array([0.,0.,1.]), z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        R = np.array([x_axis, y_axis, z_axis]).T

        tcp = np.eye(4)
        tcp[:3, :3] = R
        tcp[:3, 3] = position

        return tcp
