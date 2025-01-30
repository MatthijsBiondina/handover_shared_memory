import argparse
import numpy as np
import os
import time
#from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85

from sensor_comm_dds.data_logging.data_logger import IRTouchDataLogger
from sensor_comm_dds.utils.paths import data_logging_path


class Robotiq2F85Mock:
    def __init__(self, host_ip):
        pass

    def _activate_gripper(self):
        pass

    def move(self, _, __, ___):
        class Awaitable:
            def wait(self):
                pass
        return Awaitable()


class CalibrationDataHandler(IRTouchDataLogger):
    def __init__(self, topic_name, directory=os.path.join(data_logging_path, 'irtouch32/calibration'), buffer_size=1):
        super().__init__(topic_name=topic_name, directory=directory, buffer_size=buffer_size)
        self.csv_header += ['gripper_position']
        self.init_csv_file(self.directory, self.file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This script reads IRTouch32 sensor values for different gripper widths and logs them, so that"
                    "they can be used as calibration values for the irtouch32_viewmodel. Note that an irtouch32_reader"
                    "must be running and publishing to a topic 'topic_name'.")
    parser.add_argument('topic_name', type=str, help='Name of the topic where the sensor data is published')
    parser.add_argument('host_ip', nargs='?', type=str, help='IP address of the robot.')
    args = parser.parse_args()
    topic_name = args.topic_name
    host_ip = args.host_ip
    if not host_ip:
        host_ip = "10.42.0.162"
    gripper = Robotiq2F85Mock(host_ip)
    gripper._activate_gripper()
    time.sleep(5)

    data_handler = CalibrationDataHandler(topic_name=topic_name, buffer_size=0)
    speed = 150
    force = 150
    widths_to_evaluate = np.linspace(start=0.015, stop=0.004, num=10)
    data = {}
    num_avgs = 3
    for width in widths_to_evaluate:
        data[width] = []
        for _ in range(num_avgs):
            print(f'{width} iteration {_}')
            gripper.move(0.085, speed, force).wait()
            gripper.move(width, speed, force).wait()
            time.sleep(0.5)
            data[width].append(data_handler.read_single_sample())
    for i, width in enumerate(widths_to_evaluate):
        data[width] = np.array(data[width])
        avg_data = np.mean(data[width], 0)
        data_handler.persist(list(avg_data) + [width])

    gripper.move(0.085, speed, force).wait()
