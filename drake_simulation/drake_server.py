from cantrips.logging.logger import get_logger

import logging
import os
import time
from pathlib import Path

import numpy as np

time.sleep(1)
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelConfig
from curobo.types.robot import RobotConfig
from cyclonedds.domain import DomainParticipant
from munch import Munch

from cantrips.configs import load_config
from cantrips.exceptions import WaitingForFirstMessageException

from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.curobo.collision_spheres_sample import CuroboCollisionSpheresSample
from cyclone.idl.simulation.world_config_rpc_sample import WorldConfigRPC
from cyclone.idl.ur5e.joint_configuration_sample import JointConfigurationSample
from cyclone.idl.ur5e.tcp_pose_sample import TCPPoseSample
from cyclone.patterns.responder import Responder
from cyclone.patterns.ddsreader import DDSReader
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from drake_simulation.drake_scene import DrakeScene


logger = get_logger("INFO")
logger.info("Drake Server Imports Finished!")


class Readers:
    def __init__(self, participant: DomainParticipant, config: Munch):
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
        self.goal_pose = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.TARGET_TCP_POSE,
            idl_dataclass=TCPPoseSample,
        )
        self.curobo_spheres = DDSReader(
            domain_participant=participant,
            topic_name=CYCLONE_NAMESPACE.CUROBO_COLLISION_SPHERES,
            idl_dataclass=CuroboCollisionSpheresSample,
        )


class Writers:
    def __init__(self, participant: DomainParticipant, config: Munch):
        pass


class DrakeServer:
    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()

        # Simulator Setup
        spheres = self.__init_spheres()
        self.simulator = DrakeScene(spheres)

        # Cyclone Setup
        self.participant = participant
        self.readers = Readers(participant, self.config)
        self.responder = Responder(
            domain_participant=self.participant,
            rpc_name=CYCLONE_NAMESPACE.WORLD_CONFIG,
            idl_dataclass=WorldConfigRPC,
            callback=self.__get_world_config,
        )

        logger.info(f"DrakeServer: Ready!")

    def run(self):
        while True:
            try:
                self.__update_joint_configuration()
                self.__update_tcp_pose()
                self.__update_curobo_spheres()
                self.__update_goal_pose()
            except WaitingForFirstMessageException:
                pass
            self.participant.sleep()

    def __init_spheres(self):
        # Sphere setup
        robot_config_yaml = (
            Path(os.path.dirname(__file__))
            / "../curobo_simulation/content/configs/robot/ur5e_robotiq_2f_85.yml"
        )
        robot_config = RobotConfig(
            CudaRobotModelConfig.from_robot_yaml_file(str(robot_config_yaml))
        )
        spheres = robot_config.kinematics.kinematics_config.link_spheres
        return spheres

    def __update_joint_configuration(self):
        joint_configuration_sample = self.readers.joint_configuration()

        if joint_configuration_sample is None:
            raise WaitingForFirstMessageException
        self.simulator.joint_state = (
            np.array(joint_configuration_sample.pose),
            np.array(joint_configuration_sample.velocity),
        )

    def __update_curobo_spheres(self):
        spheres = self.readers.curobo_spheres()
        self.simulator.update_spheres(spheres)

    def __update_tcp_pose(self):
        tcp_pose_sample = self.readers.tcp_pose()
        if tcp_pose_sample is None:
            raise WaitingForFirstMessageException
        self.simulator.tcp_pose = np.array(tcp_pose_sample.pose)

    def __update_goal_pose(self):
        goal_pose_sample = self.readers.goal_pose()
        if goal_pose_sample is None:
            raise WaitingForFirstMessageException
        self.simulator.goal_pose = np.array(goal_pose_sample.pose)

    def __get_world_config(
        self, request: WorldConfigRPC.Request
    ) -> WorldConfigRPC.Response:
        world_config = self.simulator.world_config
        cuboid = world_config.cuboid
        response = WorldConfigRPC.Response(
            timestamp=request.timestamp,
            cuboid=list(cuboid.keys()),
            cuboid_dims=[cuboid[key]["dims"].tolist() for key in cuboid],
            cuboid_pose=[cuboid[key]["pose"].tolist() for key in cuboid],
        )
        return response


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = DrakeServer(participant)
    node.run()
