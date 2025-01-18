import os
import time
import traceback
from pathlib import Path

import yaml
from curobo.cuda_robot_model.cuda_robot_model import (
    CudaRobotModelConfig,
    CudaRobotModel,
)
from curobo.types.robot import RobotConfig

from cantrips.configs import load_config
from cantrips.debugging.terminal import pyout
from cantrips.logging.logger import get_logger
from cyclone.cyclone_participant import CycloneParticipant
from drake_simulation.drake_client import DrakeClient

logger = get_logger()


def load_kinematics():
    config = load_config()
    robot_config_yaml_path = (
        f"{os.path.dirname(__file__)}/{config.robot_config_relative_path}"
    )
    cuda_robot_model_config = CudaRobotModelConfig.from_robot_yaml_file(
        str(robot_config_yaml_path)
    )
    cuda_robot_model = CudaRobotModel(cuda_robot_model_config)
    return cuda_robot_model


def load_robot_config():
    config = load_config()

    robot_config_yaml_path = (
        Path(os.path.dirname(__file__)) / config.robot_config_relative_path
    )
    cuda_robot_model_config = CudaRobotModelConfig.from_robot_yaml_file(
        str(robot_config_yaml_path)
    )
    robot_config = RobotConfig(cuda_robot_model_config)

    return robot_config


def load_base_config():
    config = load_config()

    base_config_yaml_path = Path(os.path.dirname(__file__)) / "configs" / "base_cfg.yml"

    with open(base_config_yaml_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    return base_cfg


def load_world_config(participant: CycloneParticipant):
    drake_client = DrakeClient(participant)
    for _ in range(5):
        try:
            time.sleep(1)
            return drake_client.world_config
        except RuntimeError as e:
            pass
    raise RuntimeError("Is Drake Simulator running?")
