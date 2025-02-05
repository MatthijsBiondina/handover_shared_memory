import time
import warnings

import numpy as np
from cyclonedds.domain import DomainParticipant
from munch import Munch
from airo_robots.manipulators import URrtde
from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85

from cantrips.configs import load_config
from cantrips.exceptions import WaitingForFirstMessageException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.ur5e.gripper_width_sample import GripperWidthSample
from cyclone.idl.ur5e.joint_configuration_sample import JointConfigurationSample
from cyclone.idl.ur5e.trajectory_sample import TrajectorySample
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.ddswriter import DDSWriter
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
        self.gripper = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.TARGET_GRIPPER_WIDTH,
            idl_dataclass=GripperWidthSample,
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
        self.sophie = URrtde(self.config.ip_sophie, URrtde.UR3E_CONFIG)
        self.sophie.gripper = Robotiq2F85(self.config.ip_sophie)

        self.sophie.gripper.move(0.05).wait()
        self.sophie.gripper.open().wait()
        

        # Cyclone setup
        self.participant = participant
        self.readers = Readers(participant)
        self.writers = Writers(participant)

        # Placeholders
        self.__timestamp = None
        self.__joint_configuration = None
        self.__tcp_pose = None
        self.__trajectory = None
        self.__gripper_action = None
        self.__gripper_tgt_width = None
        self.is_holding_object = False

        logger.info("UR5eRobotArm: Ready!")

    def run(self):
        ii = 0
        while True:
            try:
                self.__publish_robot_state()
                action = self.__get_action()
                self.sophie.servo_to_joint_configuration(action, self.config.dt)
                self.__handle_gripper_width()

            except WaitingForFirstMessageException:
                pass
            self.participant.sleep()

    def __handle_gripper_width(self):
        current_width: float = self.sophie.gripper.get_current_width()
        gripper_tgt: GripperWidthSample = self.readers.gripper.take()
        if gripper_tgt is None:
            if self.__gripper_action is None:
                return

            if np.isclose(current_width, self.__gripper_tgt_width, atol=0.001):
                self.__cancel_gripper_action()
                return
            if (
                self.sophie.gripper.is_an_object_grasped()
                and self.__gripper_tgt_width < current_width
            ):
                self.is_holding_object = True
                self.__cancel_gripper_action()

        else:
            try:
                target_width = gripper_tgt.width
            except AttributeError:
                return
            self.__gripper_tgt_width = target_width
            if np.isclose(current_width, target_width, atol=0.001):
                return  # If we're already at target width, do nothing
            elif target_width > current_width:
                self.__gripper_action = self.sophie.gripper.move(target_width, force=25)
                self.is_holding_object = (
                    False  # if we were holding something, we just released it
                )
            elif target_width < current_width:
                self.__gripper_action = self.sophie.gripper.move(target_width, force=25)

    def __cancel_gripper_action(self):
        if self.__gripper_action is None:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.__gripper_action.wait(0)
        self.__gripper_action = None

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
        if self.sophie.gripper is not None:
            gripper_width = self.sophie.gripper.get_current_width()
        else:
            gripper_width = 0.1
        self.writers.gripper(
            GripperWidthSample(
                timestamp=timestamp_new,
                width=gripper_width,
                holding=self.sophie.gripper.is_an_object_grasped(),
            )
        )

        # Update placeholders
        self.__timestamp = timestamp_new
        self.__joint_configuration = joint_configuration_new
        self.__tcp_pose = tcp_pose_new

    def __get_action(self):
        if self.__trajectory is not None:
            return self.__interpolate_trajectory()

        # See if there's a new trajectory
        trajectory_sample: TrajectorySample | None = self.readers.trajectory.take()
        if isinstance(trajectory_sample, TrajectorySample):
            self.__trajectory = np.array(trajectory_sample.trajectory)
            return self.__interpolate_trajectory()

        # Otherwise, see if there's an action
        action_sample: JointConfigurationSample | None = self.readers.action()
        if isinstance(action_sample, JointConfigurationSample):
            return self.__extract_action(action_sample)

        raise WaitingForFirstMessageException

    def __interpolate_trajectory(
        self,
        n_interpolation_steps: int = 10,
        n_steps_forward: int = 3,
    ):
        repeated_trajectory = np.repeat(
            self.__trajectory, n_interpolation_steps, axis=0
        )
        alpha = np.tile(
            np.linspace(0.0, 1.0, n_interpolation_steps), len(self.__trajectory) - 1
        )[:, None]
        interpolated_trajectory = (1 - alpha) * repeated_trajectory[
            :-n_interpolation_steps
        ] + alpha * repeated_trajectory[n_interpolation_steps:]

        distance_to_current_pose = np.linalg.norm(
            interpolated_trajectory - self.sophie.get_joint_configuration()[None, :],
            axis=1,
        )
        if distance_to_current_pose[-1] < 1e-3:
            self.__trajectory = None
            return interpolated_trajectory[-1]

        current_idx = np.argmin(distance_to_current_pose)
        for ii in range(n_interpolation_steps * n_steps_forward + 1):
            try:
                idx = current_idx + n_interpolation_steps * n_steps_forward - ii
                action = interpolated_trajectory[idx]
                if idx >= interpolated_trajectory.shape[0] - 1:
                    self.__trajectory = None
                return action
            except IndexError:
                pass

    def __extract_action(self, action_sample: JointConfigurationSample):
        pose = np.array(action_sample.pose)
        dpose = pose - self.sophie.get_joint_configuration()
        if np.absolute(dpose).max() > self.config.max_angular_velocity * self.config.dt:

            # return self.sophie.get_joint_configuration()

            dpose = (
                (dpose / np.absolute(dpose).max())
                * self.config.max_angular_velocity
                * self.config.dt
            )

        velocity = np.array(action_sample.velocity)

        return self.sophie.get_joint_configuration() + dpose


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = UR5eRobotArm(participant)
    node.run()
