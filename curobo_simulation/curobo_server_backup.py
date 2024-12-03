import os
import time
from gc import enable
from pathlib import Path

import numpy as np
import torch
import yaml
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.wrap.reacher.mpc import MpcSolverConfig, MpcSolver
from cyclonedds.domain import DomainParticipant
from torch import tensor

from cantrips.configs import load_config
from cantrips.debugging.terminal import pyout
from cantrips.exceptions import WaitingForFirstMessageException
from cantrips.logging.logger import get_logger, shht
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.curobo.collision_spheres_sample import CuroboCollisionSpheresSample
from cyclone.idl.ur5e.joint_configuration_sample import JointConfigurationSample
from cyclone.idl.ur5e.tcp_pose_sample import TCPPoseSample
from cyclone.shared_memory.ddsreader import DDSReader
from cyclone.shared_memory.ddswriter import DDSWriter
from drake_simulation.drake_client import DrakeClient
from utils.simulation_utils import cupose2matrix, matrix2cupose, numpy2cspace

logger = get_logger()


class Readers:
    def __init__(self, participant: DomainParticipant):
        self.joint_configuration = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.UR5E_JOINT_CONFIGURATION,
            idl_dataclass=JointConfigurationSample,
        )
        self.goal_js = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.TARGET_JOINT_CONFIGURATION,
            idl_dataclass=JointConfigurationSample,
        )
        self.goal_tcp = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.TARGET_TCP_POSE,
            idl_dataclass=TCPPoseSample,
        )


class Writers:
    def __init__(self, participant: DomainParticipant):
        self.tcp_pose = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.UR5E_TCP_POSE,
            idl_dataclass=TCPPoseSample,
        )
        self.action = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.ACTION_JOINT_CONFIGURATION,
            idl_dataclass=JointConfigurationSample,
        )
        self.spheres = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.CUROBO_COLLISION_SPHERES,
            idl_dataclass=CuroboCollisionSpheresSample,
        )


class CuroboServer:
    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()
        self.drake_client = DrakeClient(participant)
        self.world_config = self.__get_world_config()

        self.mpc: MpcSolver = self.__init_mpc_solver()

        default_goal_state = (
            tensor(np.deg2rad(self.config.default_joint_state), dtype=torch.float32)
            .cuda()
            .contiguous()
        )
        self.default_goal_state = JointState.from_position(
            position=default_goal_state, joint_names=self.config.joint_names
        )
        self.goal_buffer = self.__init_goal_buffer()

        self.participant = participant
        self.readers = Readers(participant)
        self.writers = Writers(participant)

        logger.warn("CuroboServer: Ready!")

    def __get_world_config(self):
        for _ in range(5):
            try:
                time.sleep(1)
                world_config = self.drake_client.world_config
                return world_config
            except RuntimeError:
                time.sleep(0.01)

        raise RuntimeError("Is Drake Simulator running?")

    def __init_mpc_solver(self) -> MpcSolver:
        robot_config_yaml = (
            Path(os.path.dirname(__file__))
            / "content/configs/robot/ur5e_robotiq_2f_85.yml"
        )
        robot_config = RobotConfig(
            CudaRobotModelConfig.from_robot_yaml_file(str(robot_config_yaml))
        )

        with open(Path(os.path.dirname(__file__)) / "base_cfg.yml", "r") as f:
            base_cfg = yaml.load(f, Loader=yaml.FullLoader)

        mpc_config = MpcSolverConfig.load_from_robot_config(
            robot_config,
            self.world_config.as_dictionary(),
            base_cfg=base_cfg,
            collision_activation_distance=0.03,
            particle_opt_iters=128,
            store_rollouts=True,
            step_dt=self.config.dt,
        )
        mpc = MpcSolver(mpc_config)
        return mpc

    def __init_goal_buffer(self):
        position = (
            torch.zeros((self.config.num_seeds, 6), dtype=torch.float32)
            .cuda()
            .contiguous()
        )
        velocity = torch.zeros_like(position)
        acceleration = torch.zeros_like(position)

        current_state = JointState.from_position(
            position=position,
            joint_names=self.config.joint_names,
        )
        current_state.velocity = velocity
        current_state.acceleration = acceleration

        goal_pose = matrix2cupose(
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.75],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
        goal = Goal(
            current_state=current_state,
            goal_pose=goal_pose,
            goal_state=self.default_goal_state,
        )
        goal_buffer = self.mpc.setup_solve_single(goal, num_seeds=self.config.num_seeds)
        self.mpc.update_goal(goal_buffer)
        return goal_buffer

    def run(self):
        while True:
            try:
                current_joint_state = self.__get_joint_configuration()
                self.__publish_tcp_pose(current_joint_state)
                self.__publish_collision_spheres(current_joint_state)

                # goal_pose = self.__get_goal_pose()
                goal_pose, goal_js = self.__recieve_new_goal()

                if goal_pose is not None:
                    self.__update_goal_buffer_from_homogeneous_matrix(
                        current_joint_state, goal_pose
                    )
                if goal_js is not None:
                    self.__update_goal_buffer_from_joint_state(
                        current_joint_state, goal_js
                    )

                self.__step(current_joint_state)

            except WaitingForFirstMessageException:
                pass
            self.participant.sleep()

    def __get_joint_configuration(self):
        joint_configuration_sample = self.readers.joint_configuration()
        if joint_configuration_sample is None:
            raise WaitingForFirstMessageException

        batch_size = self.config.num_seeds
        position = torch.tensor(
            joint_configuration_sample.pose, dtype=torch.float32
        ).cuda()
        position = position.unsqueeze(0).expand(batch_size, -1).contiguous()
        velocity = torch.tensor(
            joint_configuration_sample.velocity, dtype=torch.float32
        ).cuda()
        velocity = velocity.unsqueeze(0).expand(batch_size, -1).contiguous()

        joint_state = JointState.from_position(
            position,
            joint_names=self.config.joint_names,
        )
        joint_state.velocity = velocity
        joint_state.acceleration = torch.zeros_like(velocity)

        return joint_state

    def __publish_tcp_pose(self, joint_state: JointState):
        tcp_pose = self.__calculate_tcp_pose(joint_state)
        self.writers.tcp_pose(
            TCPPoseSample(
                timestamp=time.time(),
                pose=tcp_pose.tolist(),
                velocity=np.zeros_like(tcp_pose).tolist(),
            )
        )

    def __publish_collision_spheres(self, joint_state: JointState):
        spheres = self.mpc.kinematics.get_robot_as_spheres(
            joint_state.position.squeeze(0)
        )[0]

        sample = CuroboCollisionSpheresSample(
            timestamp=time.time(), positions=[sphere.position for sphere in spheres]
        )
        self.writers.spheres(sample)

    def __calculate_tcp_pose(self, joint_state: JointState):
        cupose: Pose = self.mpc.kinematics.get_state(joint_state.position[0]).ee_pose
        matrix: np.ndarray = cupose2matrix(cupose)
        return matrix

    def __recieve_new_goal(self):
        goal_tcp_sample = self.readers.goal_tcp()
        goal_js_sample = self.readers.goal_js()

        if goal_tcp_sample is None and goal_js_sample is None:
            raise WaitingForFirstMessageException
        if goal_tcp_sample is not None and goal_js_sample is None:
            return np.array(goal_tcp_sample.pose), None
        if goal_tcp_sample is None and goal_js_sample is not None:
            return None, numpy2cspace(
                np.array(goal_js_sample.pose), joint_names=self.config.joint_names
            )
        if goal_tcp_sample is not None and goal_js_sample is not None:
            if goal_tcp_sample.timestamp > goal_js_sample.timestamp:
                return np.array(goal_tcp_sample.pose), None
            else:
                return None, numpy2cspace(
                    np.array(goal_js_sample.pose), joint_names=self.config.joint_names
                )

    def __get_goal_pose(self):
        goal_tcp_sample = self.readers.goal_tcp()
        if goal_tcp_sample is None:
            raise WaitingForFirstMessageException

        goal_tcp = np.array(goal_tcp_sample.pose)
        return goal_tcp

    def __update_goal_buffer_from_homogeneous_matrix(self, current_state, goal_pose):
        self.goal_buffer.goal_pose.copy_(matrix2cupose(goal_pose))
        self.mpc.enable_pose_cost(enable=True)
        self.mpc.enable_cspace_cost(enable=False)
        self.mpc.update_goal(self.goal_buffer)

    def __update_goal_buffer_from_joint_state(
        self, current_state, goal_state: JointState
    ):
        self.goal_buffer.goal_state.copy_(goal_state)
        self.mpc.enable_pose_cost(enable=False)
        self.mpc.enable_cspace_cost(enable=True)
        self.mpc.update_goal(self.goal_buffer)

    def __step(self, joint_state):
        result = self.mpc.step(joint_state)
        action = result.action[0]
        feasible = result.metrics.feasible[0][0]
        compliant = result.metrics.constraint[0][0] == 0.0

        alpha = 0.5
        scaled_action = (
            (alpha * action.position) + (1 - alpha) * joint_state.position
        ).squeeze(0)

        if feasible and compliant:
            self.writers.action(
                JointConfigurationSample(
                    timestamp=time.time(),
                    pose=scaled_action.cpu().numpy().tolist(),
                    velocity=action.velocity.cpu().numpy().tolist(),
                )
            )


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = CuroboServer(participant)
    node.run()
