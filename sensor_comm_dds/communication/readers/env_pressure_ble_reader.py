import argparse
import struct
import time
import os
from loguru import logger
from dataclasses import dataclass
from sensor_comm_dds.communication.config.ble_config import BleConfig, SensorUuid, DeviceMAC
from sensor_comm_dds.communication.config.websocket_config import WebsocketConfig
from sensor_comm_dds.communication.data_classes.envpressure import EnvironmentPressure
from sensor_comm_dds.communication.readers.data_publisher import DataPublisher
from sensor_comm_dds.communication.readers.ble_reader import BleReader
from sensor_comm_dds.communication.readers.websocket import Websocket


@dataclass
class EnvironmentPressureConfig(WebsocketConfig, BleConfig):
    _: int = None


class EnvironmentPressureReader(BleReader):
    """
    Works for BMP384 and MS5803 sensors
    """
    def __init__(self, config: EnvironmentPressureConfig):
        super().__init__(config=config)
        self.data_publisher = DataPublisher(topic_name="EnvPressure", topic_data_type=EnvironmentPressure)
        self.websocket = Websocket(websocket_server_url=self.config.websocket_server_url)

    @staticmethod
    def data_convert(data):
        """
        Data is float (32bit), received as bytearray.
        """
        data = bytearray(data)
        pressure = struct.unpack('f', (data[:4]))[0]
        temperature = struct.unpack('f', (data[4:]))[0]
        data_converted = [pressure, temperature]
        return data_converted

    async def data_callback(self, handle: int, data: bytearray):
        data_bytes = list(data)
        data_converted = self.data_convert(data_bytes)
        logger.debug([ '%.2f' % elem for elem in data_converted ])
        env_pressure = EnvironmentPressure(pressure=data_converted[0], temperature=data_converted[1])
        self.data_publisher.publish_sensor_data(env_pressure)
        if self.config.ENABLE_WS:
            row = {f"Pressure": data_converted[0], f"Temperature": data_converted[1], "t": time.time()}
            await self.websocket.async_send_data_to_websocket(row)


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    parser = argparse.ArgumentParser(description='Read data from a BMP384 or MS5803 sensor over BLE.'
                                                 'The data will be published to topic EnvPressure')

    reader = EnvironmentPressureReader(config=EnvironmentPressureConfig(uuid=SensorUuid.DATA_CHAR_PRESSURE,
                                                                        device_mac=DeviceMAC.PCBX2,
                                                                        ENABLE_WS=True,
                                                                        hci="hci0"))
    reader.run()
