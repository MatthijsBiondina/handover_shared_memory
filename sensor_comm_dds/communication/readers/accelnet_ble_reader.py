import argparse
import time
import os
import numpy as np
from loguru import logger
from dataclasses import dataclass
from sensor_comm_dds.communication.config.ble_config import BleConfig, SensorUuid, DeviceMAC
from sensor_comm_dds.communication.config.websocket_config import WebsocketConfig
from sensor_comm_dds.communication.data_classes.sequence import Sequence
from sensor_comm_dds.communication.readers.data_publisher import DataPublisher
from sensor_comm_dds.communication.readers.ble_reader import BleReader
from sensor_comm_dds.communication.readers.websocket import Websocket
from sensor_comm_dds.utils.conversions import interpret_16b_as_twos_complement


@dataclass
class AccelNetConfig(WebsocketConfig, BleConfig):
    NUM_ACCELS: int = 1


class AccelNetBLEReader(BleReader):
    def __init__(self, config: AccelNetConfig):
        super().__init__(config=config)
        self.data_publisher = DataPublisher(topic_name="AccelNet", topic_data_type=Sequence)
        self.websocket = Websocket(websocket_server_url=self.config.websocket_server_url)
        self.num_accels = config.NUM_ACCELS

    def data_convert(self, data):
        """
        Data is float (32bit), received as bytearray.
        """
        assert len(data) % 2 == 0, 'Data uneven length, cannot convert bytes to 16b integers'
        data_converted = np.array([interpret_16b_as_twos_complement(data[2*i] + (data[2*i + 1] << 8)) for i in range(len(data)//2)])
        data_reshaped = data_converted.reshape((self.num_accels, 3))
        return list(data_converted)

    async def data_callback(self, handle: int, data: bytearray):
        data_bytes = list(data)
        data_converted = self.data_convert(data_bytes)
        logger.debug(data_converted)
        accelnet = Sequence(values=data_converted)
        self.data_publisher.publish_sensor_data(accelnet)
        if self.config.ENABLE_WS:
            row = {f"Pressure": data_converted[0], f"Temperature": data_converted[1], "t": time.time()}
            await self.websocket.async_send_data_to_websocket(row)


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    parser = argparse.ArgumentParser(description='Read data from an AccelNet over BLE.'
                                                 'The data will be published to topic AccelNet')
    args = parser.parse_args()

    reader = AccelNetBLEReader(config=AccelNetConfig(
        NUM_ACCELS=1,
        uuid=SensorUuid.DATA_CHAR_ACCELNET,
        device_mac=DeviceMAC.ARDUINO4,
        ENABLE_WS=False,
        hci="hci0"))
    reader.run()
