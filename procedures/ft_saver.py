import argparse
import dill
import numpy as np
import os
import serial
import time
from loguru import logger
import matplotlib.pyplot as plt

from cyclone.cyclone_participant import CycloneParticipant
from sensor_comm_dds.communication.config.cyclone_config import CycloneConfig
from sensor_comm_dds.communication.readers.websocket import Websocket
from sensor_comm_dds.communication.data_classes.sequence import Sequence
from sensor_comm_dds.communication.config.serial_config import SerialConfig
from sensor_comm_dds.communication.config.websocket_config import WebsocketConfig
from dataclasses import dataclass
from sensor_comm_dds.communication.readers.data_publisher import DataPublisher
from rtde_receive import RTDEReceiveInterface


@dataclass
class FTReaderConfig(WebsocketConfig, CycloneConfig):
    ROBOT_IP = "10.42.0.162"

class FTReader:
    def __init__(self, config: FTReaderConfig):
        self.participant = CycloneParticipant(rate_hz=30)
        self.config = config
        self.websocket = Websocket(websocket_server_url=self.config.websocket_server_url)
        max_connection_attempts = 3
        for connection_attempt in range(max_connection_attempts):
            try:
                self.rtde_receive = RTDEReceiveInterface(self.config.ROBOT_IP)
                break
            except RuntimeError as e:
                logger.warning("Failed to connect to RTDE, retrying...")
                if connection_attempt == max_connection_attempts:
                    raise RuntimeError("Could not connect to RTDE. Is the robot in remote control? Is the IP correct? Is the network ok?")
                else:
                    time.sleep(1)


    def run(self):
        sample = Sequence([0 for _ in range(6)])
        while True:
            sample.values = self.rtde_receive.getActualTCPForce()
            with open(f"/home/matt/TF/{time.time():.2f}.csv", "w+") as f:
                f.write(" ".join(map(str, sample.values)))
            self.participant.sleep()

if __name__ == "__main__":
    while True:
        try:
            logger.info(f"Running {os.path.basename(__file__)}")
            parser = argparse.ArgumentParser(description='Read data from a UR F/T sensor over a LAN.')

            ft_reader = FTReader(config=FTReaderConfig(ENABLE_WS=False,
                                                    topic_names=np.array(["FT"])))

            ft_reader.run()
        except KeyboardInterrupt:
            break
        except Exception as e:
            time.sleep(0.1)
