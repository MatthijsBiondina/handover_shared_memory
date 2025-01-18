import time
from pydrake.geometry import Box  # DO NOT REMOVE!
import numpy as np
import torch
from curobo.cuda_robot_model.cuda_robot_model import (
    CudaRobotModel,
)
from curobo.rollout.rollout_base import Goal
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGenConfig, MotionGen
from curobo.wrap.reacher.mpc import MpcSolverConfig, MpcSolver
from cyclonedds.domain import DomainParticipant
from torch import tensor

from cantrips.configs import load_config
from cantrips.exceptions import WaitingForFirstMessageException
from cantrips.logging.logger import get_logger, shht

from curobo_simulation.curobo_utils import load_kinematics
from curobo_simulation.curobo_utils import load_robot_config
from curobo_simulation.curobo_utils import load_world_config
from curobo_simulation.curobo_utils import load_base_config

from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.curobo.collision_spheres_sample import CuroboCollisionSpheresSample
from cyclone.idl.ur5e.joint_configuration_sample import JointConfigurationSample
from cyclone.idl.ur5e.tcp_pose_sample import TCPPoseSample
from cyclone.idl.ur5e.trajectory_sample import TrajectorySample
from cyclone.patterns.ddsreader import DDSReader
from cyclone.patterns.ddswriter import DDSWriter
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
        self.trajectory = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.CUROBO_TRAJECTORY,
            idl_dataclass=TrajectorySample,
        )
        self.goal_tcp = DDSWriter(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.TARGET_TCP_POSE,
            idl_dataclass=TCPPoseSample,
        )


class CuroboServer:
    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()
        self.robot_config = load_robot_config()
        self.kinematics: CudaRobotModel = load_kinematics()
        self.world_config = load_world_config(participant)

        self.participant = participant
        self.readers = Readers(participant)
        self.writers = Writers(participant)

        self.current_state_placeholder = np.empty(6)

    def step(self):
        current_joint_state = self.get_joint_state()
        self.current_state_placeholder = current_joint_state.position.cpu().numpy()[0]
        # self.publish_tcp_pose(current_joint_state)
        self.__publish_collision_spheres(current_joint_state)
        return current_joint_state

    def get_goal(self):
        goal_pose = self.readers.goal_tcp()
        goal_state = self.readers.goal_js()

        if goal_pose is None and goal_state is None:
            raise WaitingForFirstMessageException()
        if goal_pose is not None and goal_state is None:
            return {"pose": matrix2cupose(np.array(goal_pose.pose))}
        if goal_pose is None and goal_state is not None:
            return {"state": numpy2cspace(np.array(goal_state.pose))}
        if goal_pose is not None and goal_state is not None:
            if goal_pose.timestamp > goal_state.timestamp:
                return {"pose": matrix2cupose(np.array(goal_pose.pose))}
            else:
                return {"state": numpy2cspace(np.array(goal_state.pose))}

    def publish_action(self, action: JointState):
        self.writers.action(
            JointConfigurationSample(
                timestamp=time.time(),
                pose=action.position.squeeze(0).cpu().numpy().tolist(),
                velocity=action.velocity.squeeze(0).cpu().numpy().tolist(),
            )
        )

    def publish_trajectory(self, trajectory: np.ndarray):
        self.writers.trajectory(
            TrajectorySample(timestamp=time.time(), trajectory=trajectory.tolist())
        )

    def get_current_state(self) -> JointState:
        joint_cfg_sample: JointConfigurationSample = self.readers.joint_configuration()
        if joint_cfg_sample is None:
            raise WaitingForFirstMessageException

        joint_state = JointState.from_numpy(
            joint_names=self.config.joint_names,
            position=np.array(joint_cfg_sample.pose)[None, :],
            velocity=np.array(joint_cfg_sample.velocity)[None, :],
        )
        return joint_state

    def get_joint_state(self):
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

    def publish_tcp_pose(self, joint_state: JointState):
        tcp_pose = self.__calculate_tcp_pose(joint_state)
        self.writers.tcp_pose(
            TCPPoseSample(
                timestamp=time.time(),
                pose=tcp_pose.tolist(),
                velocity=np.zeros_like(tcp_pose).tolist(),
            )
        )

    def __publish_collision_spheres(self, joint_state: JointState):
        spheres = self.kinematics.get_robot_as_spheres(joint_state.position.squeeze(0))[
            0
        ]

        sample = CuroboCollisionSpheresSample(
            timestamp=time.time(), positions=[sphere.position for sphere in spheres]
        )
        self.writers.spheres(sample)

    def __calculate_tcp_pose(self, joint_state: JointState):
        cupose: Pose = self.kinematics.get_state(joint_state.position[0]).ee_pose
        matrix: np.ndarray = cupose2matrix(cupose)
        return matrix

    def init_mpc(self):
        retract_state = JointState.from_position(
            tensor(np.deg2rad(self.config.retract_cfg), dtype=torch.float32)[
                None, :
            ].cuda(),
            joint_names=self.config.joint_names,
        )
        goal = Goal(
            goal_state=retract_state,
            goal_pose=self.kinematics.compute_kinematics_from_joint_state(
                retract_state
            ).ee_pose,
            current_state=retract_state,
            retract_state=retract_state.position,
        )
        mpc_config = MpcSolverConfig.load_from_robot_config(
            self.robot_config,
            self.world_config.as_dictionary(),
            load_base_config(),
            use_cuda_graph=True,
            particle_opt_iters=128,
            collision_activation_distance=0.001,
            step_dt=self.config.dt,
        )
        mpc_solver = MpcSolver(mpc_config)
        mpc_solver.enable_pose_cost(True)
        goal_buffer = mpc_solver.setup_solve_single(
            goal=goal, num_seeds=self.config.num_seeds
        )
        return mpc_solver, goal_buffer

    def init_motion_gen(self):
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_config,
            self.world_config.as_dictionary(),
            interpolation_dt=self.config.dt,
            use_cuda_graph=True,
        )
        motion_gen = MotionGen(motion_gen_config)
        motion_gen.optimize_dt = False
        motion_gen.warmup()
        return motion_gen


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = CuroboServer(participant)
    node.step()
