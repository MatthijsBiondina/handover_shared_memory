import os
import asyncio
from loguru import logger
from dataclasses import dataclass
import argparse
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from cyclonedds.idl.types import uint8
from sensor_comm_dds.utils.liveliness_listener import LivelinessListener
import time

from sensor_comm_dds.communication.config.ble_config import BleConfig, SensorUuid, DeviceMAC
from sensor_comm_dds.communication.config.websocket_config import WebsocketConfig
from sensor_comm_dds.communication.data_classes.sequence import Sequence
from sensor_comm_dds.communication.data_classes.publishable_integer import PublishableInteger
from sensor_comm_dds.communication.readers.data_publisher import DataPublisher
from sensor_comm_dds.communication.readers.ble_reader import BleReader
from sensor_comm_dds.communication.readers.websocket import Websocket


@dataclass
class SwitchesConfig(WebsocketConfig, BleConfig):
    NUM_SWITCHES: int = 2


class SwitchesBLEReader(BleReader):
    def __init__(self, config: SwitchesConfig):
        super().__init__(config=config)
        self.data_publisher = DataPublisher(topic_name="Switches", topic_data_type=Sequence)

        listener = LivelinessListener(topic_name="ToggleLEDsCmd")
        domain_participant = DomainParticipant()
        topic = Topic(domain_participant, "ToggleLEDsCmd", data_type=PublishableInteger)
        self.cmd_reader = DataReader(domain_participant, topic, listener=listener)
        self.websocket = Websocket(websocket_server_url=self.config.websocket_server_url)

    async def data_callback(self, handle: int, data: bytearray):
        _data = list(data)
        # logger.debug(_data)
        switch = Sequence(values=_data)
        self.data_publisher.publish_sensor_data(switch)
        if self.config.ENABLE_WS:
            row = {f"Switch{i}": _data[i] for i in range(self.config.NUM_SWITCHES)}
            row["t"] = time.time()
            await self.websocket.async_send_data_to_websocket(row)

    async def write_device(self, data):
        """
        Perform a single write to a characteristic
        :return: TODO allow for device to respond upon write
        """
        logger.debug("Attempting write to " + self.device_mac)
        await self.connection.write_gatt_char(self.data_char_uuid, data, response=False)

    async def async_run(self):
        """
        subscribes to notifications on the data characteristic published by the sensor
        """
        while True:
            if self.disconnected_event.is_set():
                await self.connect()
                await self.subscribe()
                self.disconnected_event.clear()
            try:
                if self.cmd_reader.read_one(timeout=1).value == 1:
                    await self.write_device(0)
            except StopIteration:
                pass
            await asyncio.sleep(1)


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    parser = argparse.ArgumentParser(description='Read data from a Switches over BLE.'
                                                 'The data will be published to topic Switches')
    args = parser.parse_args()

    reader = SwitchesBLEReader(config=SwitchesConfig(
        NUM_SWITCHES=2,
        uuid=SensorUuid.DATA_CHAR_SWITCH,
        device_mac=DeviceMAC.ARDUINO4,
        ENABLE_WS=False,
        hci="hci0"))
    reader.run()
