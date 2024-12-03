import time

from airo_robots.grippers import Robotiq2F85
from airo_robots.manipulators import URrtde
from munch import Munch

from cantrips.configs import load_config
from cantrips.logging.logger import get_logger

logger = get_logger()


class ServoTest:
    def __init__(self):
        self.config: Munch = load_config()
        self.sophie = URrtde(self.config.ip_sophie, URrtde.UR3E_CONFIG)
        self.sophie.gripper = Robotiq2F85(self.config.ip_sophie)
        logger.warning("ServoTest Ready")

    def run(self):
        ii = 0
        while True:
            logger.warning(ii)
            ii +=1
            current_position = self.sophie.get_joint_configuration()
            self.sophie.servo_to_joint_configuration(current_position, 0.01).wait()
            time.sleep(0.01)


if __name__ == "__main__":
    node = ServoTest()
    node.run()
