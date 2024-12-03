import time

import numpy as np
import torch
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.ik_solver import IKSolverConfig, IKSolver
from curobo.wrap.reacher.motion_gen import (
    MotionGenConfig,
    MotionGen,
    MotionGenPlanConfig,
)
from functorch.dim import use_c
from torch import tensor, Tensor
from trimesh.path.polygons import projected
from yourdfpy import Joint

from cantrips.debugging.terminal import pyout
from cantrips.exceptions import WaitingForFirstMessageException
from cantrips.logging.logger import get_logger
from curobo_simulation.curobo_server import CuroboServer
from cyclone.cyclone_participant import CycloneParticipant

logger = get_logger()


class CuroboMotionGen(CuroboServer):
    def __init__(self, participant: CycloneParticipant):
        super(CuroboMotionGen, self).__init__(participant)

        self.default_pose: Tensor = (
            tensor(np.deg2rad(self.config.default_joint_state), dtype=torch.float32)
            .cuda()
            .contiguous()
        )[None, :]

        self.ik_solver = self.__init_ik_solver()
        self.motion_gen = self.__init_motion_gen()

        self.traj_placeholder = None
        logger.warning("CuroboMotionGen: Ready!")

    def run(self):
        while True:
            try:
                current_state = self.step()
                goal = self.get_goal()
                if "pose" in goal:
                    self.__plan_to_pose(current_state, goal["pose"])
            except WaitingForFirstMessageException:
                pass
            finally:
                self.participant.sleep()

    def __init_motion_gen(self):
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

    def __init_ik_solver(self):
        ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_config,
            self.world_config.as_dictionary(),
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            use_cuda_graph=True,
        )
        return IKSolver(ik_config)

    def __plan_to_pose(self, current_state: JointState, pose: Pose):
        t0 = time.time()
        ik_solutions = self.ik_solver.solve_single(
            goal_pose=pose,
            retract_config=self.default_pose,
            seed_config=self.default_pose[:, None, :],
        ).solution[0]
        js_goal = JointState.from_position(
            ik_solutions, joint_names=self.config.joint_names
        )
        result = self.motion_gen.plan_single_js(current_state, js_goal)
        logger.warning(time.time() - t0)
        if time.time() - t0 < self.config.max_planning_time:
            plan = result.get_interpolated_plan()
            while len(plan.shape) > 2:
                plan = plan[0]
            self.publish_trajectory(plan.position.cpu().numpy())


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = CuroboMotionGen(participant)
    node.run()
