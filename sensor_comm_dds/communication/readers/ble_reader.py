import asyncio
import atexit
from loguru import logger
from bleak import BleakClient
from bleak.exc import BleakDBusError, BleakError


class BleReader:
    def __init__(self, config):
        self.config = config
        self.device_mac = self.config.device_mac.value
        self.data_char_uuid = self.config.uuid.value

        self.disconnected_event = asyncio.Event()
        self.disconnected_event.set()

        def disconnected_callback(_):
            logger.error(f'Lost connection from {self.device_mac}, setting disconnected_event')
            self.disconnected_event.set()

        self.connection = BleakClient(self.device_mac, timeout=10, device=self.config.hci,
                                      disconnected_callback=disconnected_callback)

        @atexit.register
        def _cleanup():
            # This function cannot have "self" as an argument and is hence defined here in __init__
            loop = asyncio.get_event_loop()
            loop.run_until_complete(__cleanup())

        async def __cleanup():
            logger.warning("Interrupted, disconnecting devices")
            await self.disconnect()

    async def connect(self):
        if self.connection.is_connected:
            logger.info(f"Tried to connect to {self.device_mac} but already connected.")
            return
        success = False
        while not success:
            #try:
            logger.debug("Connecting to " + self.device_mac)
            success = await self.connection.connect()
            #except (asyncio.TimeoutError, BleakDBusError, BleakError, EOFError) as e:
            #    logger.error(f'Connection failed: {e} (-> if blank probably asyncio.TimeoutError)')
            #    await asyncio.sleep(0.5)

        logger.debug("Connected to " + self.device_mac)

    async def disconnect(self):
        await self.connection.disconnect()
        logger.debug("Disconnected from " + self.device_mac)

    def data_callback(self, handle: int, data: bytearray):
        raise NotImplementedError

    async def subscribe(self, callback=None):
        logger.debug("Subscribing to " + self.device_mac)
        logger.debug(f'{self.device_mac} services: {self.connection.services.services}')
        if not callback:
            await self.connection.start_notify(self.data_char_uuid, self.data_callback)
        else:
            await self.connection.start_notify(self.data_char_uuid, callback)
        logger.debug("Subscribed to " + self.device_mac)

    async def unsubscribe(self):
        await self.connection.stop_notify(self.data_char_uuid)
        logger.debug("Unsubscribed from " + self.device_mac)

    async def read_device(self):
        """
        Perform a single read instead of subscribing to a characteristic
        :return: single sample
        """
        logger.debug("Attempting read from " + self.device_mac)
        data = await self.connection.read_gatt_char(self.data_char_uuid)
        await self.data_callback(handle=0, data=data)
        return list(data)

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
            await asyncio.sleep(1)

    def run(self):
        asyncio.get_event_loop().run_until_complete(self.async_run())
