import argparse
import os
import time

from dataclasses import dataclass

import numpy as np
from loguru import logger
from sensor_comm_dds.communication.config.ble_config import BleConfig, SensorUuid, DeviceMAC
from sensor_comm_dds.communication.config.websocket_config import WebsocketConfig
from sensor_comm_dds.communication.data_classes.smarttextile import SmartTextile
from sensor_comm_dds.communication.readers.data_publisher import DataPublisher
from sensor_comm_dds.communication.readers.ble_reader import BleReader
from sensor_comm_dds.communication.readers.websocket import Websocket


@dataclass
class SmartTextileConfig(WebsocketConfig, BleConfig):
    NUM_TAXELS: int = 49


class SmartTextileReader(BleReader):
    def __init__(self, config: SmartTextileConfig):
        super().__init__(config=config)
        self.num_taxels = config.NUM_TAXELS
        self.data_publisher = DataPublisher(topic_name="SmartTextile", topic_data_type=SmartTextile)
        self.websocket = Websocket(websocket_server_url=self.config.websocket_server_url)

    def data_convert(self, data):
        """
        Data is 8bit, received as bytearray (so no conversion to 16bit).
        """
        return data

    async def data_callback(self, handle: int, data: bytearray):
        data_bytes = list(data)
        data_converted = self.data_convert(data_bytes)
        # logger.debug(data_converted)
        smarttextile = SmartTextile(data_converted)
        self.data_publisher.publish_sensor_data(smarttextile)
        if self.config.ENABLE_WS:
            row = {f"Taxel{i}": data_converted[i] for i in range(len(data_converted))}
            row["t"] = time.time()
            await self.websocket.async_send_data_to_websocket(row)


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    parser = argparse.ArgumentParser(description='Read data from a SmartTextile sensor over BLE.'
                                                 'The data will be published to topic SmartTextile')
    args = parser.parse_args()

    reader = SmartTextileReader(config=SmartTextileConfig(
        uuid=SensorUuid.DATA_CHAR_SMARTTEX,
        device_mac=DeviceMAC.PCB_BLACK,
        ENABLE_WS=False,
        hci="hci0"))
    reader.run()
