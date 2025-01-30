import argparse
import time
from dataclasses import dataclass
import os
from loguru import logger
from sensor_comm_dds.communication.config.ble_config import BleConfig, SensorUuid, DeviceMAC
from sensor_comm_dds.communication.config.websocket_config import WebsocketConfig
from sensor_comm_dds.communication.data_classes.irtouch32 import IRTouch32
from sensor_comm_dds.communication.readers.data_publisher import DataPublisher
from sensor_comm_dds.communication.readers.ble_reader import BleReader
from sensor_comm_dds.communication.readers.websocket import Websocket


@dataclass
class IRTouch32Config(WebsocketConfig, BleConfig):
    ENABLE_STRAIN: bool = True  # strain sensor may not be present, if not, corresponding values will be set to -1


class IRTouch32Reader(BleReader):
    def __init__(self, config: IRTouch32Config):
        """
        Cell indexing:
           _ _           _ _
         /     \\      /     \\
        /   0   \\_ _ /   5   \\
        \\      /     \\      /
         \\_ _ /   3   \\_ _ /
         /     \\      /     \\
        /   1   \\_ _ /   6   \\
        \\      /     \\      /
         \\_ _ /   4   \\_ _ /
         /     \\      /     \\
        /   2   \\_ _ /   7   \\
        \\      /     \\      /
         \\_ _ /       \\_ _ /
        """
        super().__init__(config=config)
        self.data_publisher = DataPublisher(topic_name="IRTouch32", topic_data_type=IRTouch32)
        self.websocket = Websocket(websocket_server_url=self.config.websocket_server_url)

    def data_convert(self, data):
        """
        Data is 8bit, received as bytearray (so no conversion to 16bit).
        """
        for i in range(len(data) - 1):  # final data value is strain
            data[i] = 255 - data[i]
        return data

    async def data_callback(self, handle: int, data: bytearray):
        data_bytes = list(data)
        data_converted = self.data_convert(data_bytes)
        # logger.debug(data_converted)
        if self.config.ENABLE_STRAIN:
            irtouch32 = IRTouch32(taxel_values=data_converted[:-1],
                                  strain_value=data_converted[-1])
        else:
            irtouch32 = IRTouch32(taxel_values=data_converted[:-1],
                                  strain_value=-1)
        self.data_publisher.publish_sensor_data(irtouch32)
        if self.config.ENABLE_WS:
            row = {f"Taxel{i}": data_converted[i] for i in range(len(data_converted))}
            row["Strain"] = irtouch32.strain_value
            row["t"] = time.time()
            await self.websocket.async_send_data_to_websocket(row)


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    parser = argparse.ArgumentParser(description='Read data from a IRTouch32 sensor over BLE.'
                                                 'The data will be published to topic IRTouch32')
    args = parser.parse_args()

    irtouch32_reader = IRTouch32Reader(config=IRTouch32Config(
        uuid=SensorUuid.DATA_CHAR_IRTOUCH,
        device_mac=DeviceMAC.HALBERD1,
        ENABLE_WS=False,
        hci="hci0"))
    irtouch32_reader.run()
