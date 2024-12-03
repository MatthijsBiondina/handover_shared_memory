import numpy as np
import torch
from curobo.rollout.rollout_base import Goal
from curobo.types.state import JointState
from curobo.wrap.reacher.mpc import MpcSolverConfig, MpcSolver
from torch import tensor

from cantrips.debugging.terminal import pyout
from cantrips.exceptions import WaitingForFirstMessageException
from curobo_simulation.curobo_server import CuroboServer
from curobo_simulation.curobo_utils import load_base_config
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.ur5e.tcp_pose_sample import TCPPoseSample
from utils.simulation_utils import matrix2cupose, numpy2cspace


class CuroboMPC(CuroboServer):
    def __init__(self, participant: CycloneParticipant):
        super(CuroboMPC, self).__init__(participant)
        self.retract_cfg = JointState.from_position(
            tensor(
                np.deg2rad(self.config.retract_cfg)[None, :], dtype=torch.float32
            ).cuda(),
            joint_names=self.config.joint_names,
        )
        self.mpc_solver, self.goal_buffer = self.__init_mpc()

    def run(self):
        while True:
            try:
                current_state = self.step()

                self.update_goal_buffer(current_state)

                result = self.mpc_solver.step(current_state)
                self.publish_action(result.action)
            except WaitingForFirstMessageException:
                pass
            finally:
                self.participant.sleep()

    def update_goal_buffer(self, current_state: JointState):
        goal = self.__pick_new_goal()
        self.goal_buffer.goal_pose.copy_(matrix2cupose(np.array(goal.pose)))
        self.mpc_solver.enable_cspace_cost(self.goal_buffer)

    def __pick_new_goal(self):
        goal_tcp_sample = self.readers.goal_tcp()
        if goal_tcp_sample is None:
            raise WaitingForFirstMessageException
        return goal_tcp_sample

    def __init_mpc(self):
        # 1. Create a :py:class:`~curobo.rollout.rollout_base.Goal` object with the
        # target pose or joint configuration.
        pose = self.kinematics.compute_kinematics_from_joint_state(
            self.retract_cfg
        ).ee_pose
        goal = Goal(
            goal_state=self.retract_cfg,
            goal_pose=pose,
            current_state=self.retract_cfg,
            retract_state=self.retract_cfg.position,
        )

        # 2. Create a goal buffer for the problem type using :meth:`setup_solve_single`,
        # :meth:`setup_solve_goalset`, :meth:`setup_solve_batch`, :meth:`setup_solve_batch_goalset`,
        # :meth:`setup_solve_batch_env`, or :meth:`setup_solve_batch_env_goalset`. Pass the goal
        # object from the previous step to this function. This function will update the internal
        # solve state of MPC and also the goal for MPC. An augmented goal buffer is returned.
        mpc_config = MpcSolverConfig.load_from_robot_config(
            self.robot_config,
            self.world_config.as_dictionary(),
            load_base_config(),
            use_cuda_graph=True,
            particle_opt_iters=128,
            collision_activation_distance=0.01,
            step_dt=self.config.dt,
        )
        mpc_solver = MpcSolver(mpc_config)
        mpc_solver.enable_cspace_cost(enable=True)
        mpc_solver.enable_pose_cost(enable=False)
        goal_buffer = mpc_solver.setup_solve_single(
            goal=goal, num_seeds=self.config.num_seeds
        )

        return mpc_solver, goal_buffer


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = CuroboMPC(participant)
    node.run()
