import time

import numpy as np
import torch
from curobo.types.state import JointState
from torch import tensor

from cantrips.debugging.terminal import pyout
from cantrips.exceptions import WaitingForFirstMessageException
from cantrips.logging.logger import get_logger
from curobo_simulation.curobo_server import CuroboServer
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.ur5e.joint_configuration_sample import JointConfigurationSample
from cyclone.idl.ur5e.tcp_pose_sample import TCPPoseSample
from utils.simulation_utils import cupose2matrix, matrix2cupose

logger = get_logger()


class CuroboPlanner(CuroboServer):
    def __init__(self, participant: CycloneParticipant):
        super(CuroboPlanner, self).__init__(participant)
        self.mpc_solver, self.mpc_goal_buffer = self.init_mpc()
        self.motion_gen = self.init_motion_gen()  # here it fucks up ethernet

        logger.info("CuroboPlanner: Ready!")

    def run(self):
        while True:
            try:
                current_state = self.step()
                new_goal = self.get_goal()

                if isinstance(new_goal, JointConfigurationSample):
                    self.plan_to_joint_configuration(
                        current_state, np.array(new_goal.pose)
                    )
                else:
                    self.servo_to_tcp_pose(current_state, np.array(new_goal.pose))

            except WaitingForFirstMessageException:
                pass
            except AttributeError as e:
                pass
                # logger.error(str(e))
            finally:
                self.participant.sleep()

    def get_goal(self):
        goal_js = self.readers.goal_js.take()
        if not goal_js is None:
            return goal_js

        goal_tcp = self.readers.goal_tcp()
        if goal_tcp is None:
            raise WaitingForFirstMessageException

        return goal_tcp

    def plan_to_joint_configuration(self, current_state: JointState, goal: np.ndarray):
        goal_js = JointState.from_position(
            tensor(goal, dtype=torch.float32)[None, :].cuda().contiguous(),
            joint_names=self.config.joint_names,
        )

        # Publish goal tcp for other processes
        goal_tcp = cupose2matrix(
            self.kinematics.compute_kinematics_from_joint_state(goal_js).ee_pose
        )
        self.writers.goal_tcp(
            TCPPoseSample(
                timestamp=time.time(),
                pose=goal_tcp.tolist(),
                velocity=np.zeros_like(goal_tcp).tolist(),
            )
        )

        result = self.motion_gen.plan_single_js(current_state, goal_js)
        if result.success:
            plan = result.get_interpolated_plan()
            while len(plan.shape) > 2:
                plan = plan[0]
            self.publish_trajectory(plan.position.cpu().numpy())

    def servo_to_tcp_pose(self, current_state: JointState, goal: np.ndarray):
        self.mpc_goal_buffer.goal_pose.copy_(matrix2cupose(goal))
        self.mpc_solver.update_goal(self.mpc_goal_buffer)

        seed = current_state.position[:, None, :].repeat(1, 30, 1)
        result = self.mpc_solver.step(current_state)
        self.publish_action(result.action)


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = CuroboPlanner(participant)
    node.run()
