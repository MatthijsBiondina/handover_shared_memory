from cantrips.exceptions import WaitingForFirstMessageException
from cantrips.logging.logger import get_logger
from curobo_simulation.curobo_server import CuroboServer
from cyclone.cyclone_participant import CycloneParticipant

logger = get_logger()


class CuroboForwardKinematics(CuroboServer):
    def __init__(self, participant: CycloneParticipant):
        super(CuroboForwardKinematics, self).__init__(participant)
        self.participant = CycloneParticipant(participant)

        logger.warning("Curobo Forward Kinematics: Ready!")

    def run(self):
        while True:
            try:
                joint_state = self.get_joint_state()
                self.publish_tcp_pose(joint_state)
            except WaitingForFirstMessageException:
                pass
            finally:
                self.participant.sleep()


if __name__ == "__main__":
    participant = CycloneParticipant()
    node = CuroboForwardKinematics(participant)
    node.run()
