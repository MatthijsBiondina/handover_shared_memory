import time

import numpy as np
from cyclonedds.domain import DomainParticipant
from munch import Munch
from airo_robots.manipulators import URrtde
from airo_robots.grippers import Robotiq2F85

from cantrips.configs import load_config
from cantrips.debugging.terminal import pyout
from cantrips.exceptions import WaitingForFirstMessageException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.ur5e.gripper_width_sample import GripperWidthSample
from cyclone.idl.ur5e.joint_configuration_sample import JointConfigurationSample
from cyclone.idl.ur5e.tcp_pose_sample import TCPPoseSample
from cyclone.idl.ur5e.trajectory_sample import TrajectorySample
from cyclone.shared_memory.ddsreader import DDSReader
from cyclone.shared_memory.ddswriter import DDSWriter
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE

logger = get_logger()


class Readers:
    def __init__(self, participant: DomainParticipant):
        self.action = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.ACTION_JOINT_CONFIGURATION,
            idl_dataclass=JointConfigurationSample,
        )
        self.trajectory = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.CUROBO_TRAJECTORY,
            idl_dataclass=TrajectorySample,
        )


class Writers:
    def __init__(self, participant: DomainParticipant):
        self.joint_configuration = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.UR5E_JOINT_CONFIGURATION,
            idl_dataclass=JointConfigurationSample,
        )
        self.gripper = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.UR5E_GRIPPER_WIDTH,
            idl_dataclass=GripperWidthSample,
        )


class UR5eRobotArm:
    def __init__(self, participant: CycloneParticipant):
        self.config: Munch = load_config()
        # Robot setup
        # self.wilson = URrtde(self.config.ip_wilson, URrtde.UR3E_CONFIG)
        # self.wilson.gripper = Robotiq2F85(self.config.ip_wilson)
        self.sophie = URrtde(self.config.ip_sophie, URrtde.UR3E_CONFIG)
        self.sophie.gripper = Robotiq2F85(self.config.ip_sophie)
        # self.__init_robot_configuration()

        # Cyclone setup
        # self.participant = participant
        # self.readers = Readers(participant)
        # self.writers = Writers(participant)

        # Placeholders
        # self.__timestamp = None
        # self.__joint_configuration = None
        # self.__tcp_pose = None

        logger.info("UR5eRobotArm: Ready!")

    # def __init_robot_configuration(self):
        # self.wilson.move_to_joint_configuration(
        #     np.deg2rad(self.config.wilson_rest), joint_speed=0.05
        # ).wait()
        # self.sophie.move_to_joint_configuration(
        #     np.deg2rad(self.config.default_joint_state), joint_speed=0.1
        # ).wait()

    def run(self):
        ii = 0
        while True:
            try:
                # pyout(ii)
                # ii += 1
                # self.__publish_robot_state()
                # action = self.__get_action()
                # self.sophie.servo_to_joint_configuration(action, self.config.dt).wait()
                current = self.sophie.get_joint_configuration()
                self.sophie.servo_to_joint_configuration(current, self.config.dt).wait()
            except WaitingForFirstMessageException:
                pass
                # self.sophie.servo_to_joint_configuration(
                #     self.sophie.get_joint_configuration(), self.config.dt
                # ).wait()
            self.participant.sleep()

    def __publish_robot_state(self):
        # Request data from robot
        joint_configuration_new = self.sophie.get_joint_configuration()
        tcp_pose_new = self.sophie.get_tcp_pose()
        timestamp_new = time.time()

        # Compute velocity
        joint_velocity = np.zeros_like(joint_configuration_new)
        tcp_velocity = np.zeros_like(tcp_pose_new)
        if self.__timestamp is not None:
            dt = timestamp_new - self.__timestamp
            if self.__joint_configuration is not None:
                joint_velocity = (
                    joint_configuration_new - self.__joint_configuration
                ) / dt
            if self.__tcp_pose is not None:
                tcp_velocity = (tcp_pose_new - self.__tcp_pose) / dt

        # Publish
        self.writers.joint_configuration(
            JointConfigurationSample(
                timestamp=timestamp_new,
                pose=joint_configuration_new.tolist(),
                velocity=joint_velocity.tolist(),
            )
        )
        self.writers.gripper(
            GripperWidthSample(
                timestamp=timestamp_new, width=self.sophie.gripper.get_current_width()
            )
        )

        # Update placeholders
        self.__timestamp = timestamp_new
        self.__joint_configuration = joint_configuration_new
        self.__tcp_pose = tcp_pose_new

    def __get_action(self):
        action_sample = self.readers.action()
        trajectory_sample: TrajectorySample | None = self.readers.trajectory()
        if trajectory_sample is None and action_sample is None:
            raise WaitingForFirstMessageException
        if trajectory_sample is not None and (
            action_sample is None
            or trajectory_sample.timestamp > action_sample.timestamp
        ):
            return self.__interpolate_trajectory(trajectory_sample)
        else:
            return np.array(action_sample.pose)

    def __interpolate_trajectory(
        self,
        trajectory_sample: TrajectorySample,
        n_interpolation_steps: int = 1,
        n_steps_forward: int = 1,
    ):
        t = np.array(trajectory_sample.trajectory)

        A = np.repeat(t, n_interpolation_steps, axis=0)
        alpha = np.tile(np.linspace(0.0, 1.0, n_interpolation_steps), len(t) - 1)[
            :, None
        ]
        B = (1 - alpha) * A[:-n_interpolation_steps] + alpha * A[n_interpolation_steps:]

        D = np.linalg.norm(B - self.sophie.get_joint_configuration()[None, :], axis=1)
        current_idx = np.argmin(D)

        for ii in range(n_interpolation_steps * n_steps_forward):
            try:
                return B[current_idx + n_interpolation_steps * n_steps_forward - ii]
            except IndexError:
                pass


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = UR5eRobotArm(participant)
    node.run()
