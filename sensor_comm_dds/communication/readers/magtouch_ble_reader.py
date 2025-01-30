import argparse
import asyncio
import os
import time
from dataclasses import dataclass

import dill
import numpy as np
from loguru import logger
import pandas as pd

from sensor_comm_dds.communication.config.ble_config import BleConfig, SensorUuid, DeviceMAC
from sensor_comm_dds.communication.config.websocket_config import WebsocketConfig
from sensor_comm_dds.communication.data_classes.magtouch4 import MagTouch4
from sensor_comm_dds.communication.data_classes.magtouch_taxel import MagTouchTaxel
from sensor_comm_dds.communication.readers.ble_reader import BleReader
from sensor_comm_dds.communication.readers.data_publisher import DataPublisher
from sensor_comm_dds.communication.readers.websocket import Websocket
from sensor_comm_dds.utils.paths import python_src_root


@dataclass
class MagTouchBleReaderConfig(WebsocketConfig, BleConfig):
    NUM_SENSORS: int = 2  # Number of fingertips
    NUM_TAXELS: int = 4  # Number of sensors in the array
    MODEL_NAMES: np.ndarray = np.array(["remko-2x2-003", "remko-2x2-003"])  # Names of the model to load for processing
    WINDOW_SIZE: int = 100  # Number of samples to collect before starting to predict
    SCALE_FACTOR: int = 1000  # Scale factor for the input data
    GAIN: int = 7  # Gain setting (same as firmware)
    RESOLUTION: int = 0  # Resolution setting (same as firmware)
    MAX_X = 5  # N
    MAX_Y = 5  # N
    MAX_Z = 0  # N
    MIN_X = -5  # N
    MIN_Y = -5  # N
    MIN_Z = -10  # N
    mlx90393_lsb_lookup = [
        # HALLCONF = 0xC (default)
        [
            # GAIN_SEL = 0, 5x gain
            [[0.751, 1.210], [1.502, 2.420], [3.004, 4.840], [6.009, 9.680]],
            # GAIN_SEL = 1, 4x gain
            [[0.601, 0.968], [1.202, 1.936], [2.403, 3.872], [4.840, 7.744]],
            # GAIN_SEL = 2, 3x gain
            [[0.451, 0.726], [0.901, 1.452], [1.803, 2.904], [3.605, 5.808]],
            # GAIN_SEL = 3, 2.5x gain
            [[0.376, 0.605], [0.751, 1.210], [1.502, 2.420], [3.004, 4.840]],
            # GAIN_SEL = 4, 2x gain
            [[0.300, 0.484], [0.601, 0.968], [1.202, 1.936], [2.403, 3.872]],
            # GAIN_SEL = 5, 1.667x gain
            [[0.250, 0.403], [0.501, 0.807], [1.001, 1.613], [2.003, 3.227]],
            # GAIN_SEL = 6, 1.333x gain
            [[0.200, 0.323], [0.401, 0.645], [0.801, 1.291], [1.602, 2.581]],
            # GAIN_SEL = 7, 1x gain
            [[0.150, 0.242], [0.300, 0.484], [0.601, 0.968], [1.202, 1.936]]
        ],
        # HALLCONF = 0x0
        [
            # GAIN_SEL = 0, 5x gain
            [[0.787, 1.267], [1.573, 2.534], [3.146, 5.068], [6.292, 10.137]],
            # GAIN_SEL = 1, 4x gain
            [[0.629, 1.014], [1.258, 2.027], [2.517, 4.055], [5.034, 8.109]],
            # GAIN_SEL = 2, 3x gain
            [[0.472, 0.760], [0.944, 1.521], [1.888, 3.041], [3.775, 6.082]],
            # GAIN_SEL = 3, 2.5x gain
            [[0.393, 0.634], [0.787, 1.267], [1.573, 2.534], [3.146, 5.068]],
            # GAIN_SEL = 4, 2x gain
            [[0.315, 0.507], [0.629, 1.014], [1.258, 2.027], [2.517, 4.055]],
            # GAIN_SEL = 5, 1.667x gain
            [[0.262, 0.422], [0.524, 0.845], [1.049, 1.689], [2.097, 3.379]],
            # GAIN_SEL = 6, 1.333x gain
            [[0.210, 0.338], [0.419, 0.676], [0.839, 1.352], [1.678, 2.703]],
            # GAIN_SEL = 7, 1x gain
            [[0.157, 0.253], [0.315, 0.507], [0.629, 1.014], [1.258, 2.027]]
        ]
    ]


class MagTouchBleReader(BleReader):
    def __init__(self, config: MagTouchBleReaderConfig):
        assert (config.NUM_SENSORS == len(config.MODEL_NAMES),
                f"Configured for {config.NUM_SENSORS} fingertips but passed {len(config.MODEL_NAMES)} "
                f"calibration models.")
        super().__init__(config=config)
        self.websocket = Websocket(websocket_server_url=self.config.websocket_server_url)
        self.num_sensors = self.config.NUM_SENSORS
        self.data_publishers = []
        self.raw_data_publishers = []
        for i in range(self.config.NUM_SENSORS):
            self.data_publishers.append(DataPublisher(topic_name="MagTouch" + str(i), topic_data_type=MagTouch4))
            self.raw_data_publishers.append(DataPublisher(topic_name="MagTouchRaw" + str(i), topic_data_type=MagTouch4))

        self.taxel_models = [None for _ in range(self.config.NUM_SENSORS)]
        for sensor_idx in range(self.config.NUM_SENSORS):
            if self.config.MODEL_NAMES[sensor_idx]:
                self.taxel_models[sensor_idx] = (dill.load(
                    open(os.path.join(python_src_root,
                                      f'calibration/magtouch/models/{self.config.MODEL_NAMES[sensor_idx]}'),
                         'rb')))

        self.reading_first_sample = True
        self.prev_x = [[0] * self.config.NUM_TAXELS for _ in range(self.config.NUM_SENSORS)]
        self.prev_y = [[0] * self.config.NUM_TAXELS for _ in range(self.config.NUM_SENSORS)]
        self.prev_z = [[0] * self.config.NUM_TAXELS for _ in range(self.config.NUM_SENSORS)]
        self.cycle_x = [[0] * self.config.NUM_TAXELS for _ in range(self.config.NUM_SENSORS)]
        self.cycle_y = [[0] * self.config.NUM_TAXELS for _ in range(self.config.NUM_SENSORS)]
        self.cycle_z = [[0] * self.config.NUM_TAXELS for _ in range(self.config.NUM_SENSORS)]
        self.means = np.zeros((self.config.NUM_SENSORS, self.config.NUM_TAXELS, 3))
        self.taxels = np.array([[MagTouchTaxel(x=0, y=0, z=0) for _ in range(self.config.NUM_TAXELS)]
                                for __ in range(self.config.NUM_SENSORS)])  # preallocate
        self.taxels_raw = np.array([[MagTouchTaxel(x=0, y=0, z=0) for _ in range(self.config.NUM_TAXELS)]
                                for __ in range(self.config.NUM_SENSORS)])  # preallocate
        self.magtouch_data = [MagTouch4(self.taxels[i]) for i in range(self.config.NUM_SENSORS)]
        self.magtouch_data_raw = [MagTouch4(self.taxels[i]) for i in range(self.config.NUM_SENSORS)]
        self.calibration_samples = np.zeros(
            (self.config.WINDOW_SIZE, self.config.NUM_SENSORS, self.config.NUM_TAXELS, 3))
        self.calibration_ctr = 0
        asyncio.get_event_loop().run_until_complete(self.calibrate())

    def data_convert(self, data_bytes):
        """
        Data is 8bit, received as bytearray (so no conversion to 16bit).
        """
        sample = np.zeros((self.num_sensors, self.config.NUM_TAXELS, 3))
        sample_raw = np.zeros((self.num_sensors, self.config.NUM_TAXELS, 3))
        for sensor_idx in range(self.num_sensors):
            for taxel_idx in range(self.config.NUM_TAXELS):
                offset = sensor_idx * 6 * self.config.NUM_TAXELS
                x = ~ ((data_bytes[offset + taxel_idx * 3 * 2] << 8) + data_bytes[offset + taxel_idx * 3 * 2 + 1])
                y = ~ ((data_bytes[offset + taxel_idx * 3 * 2 + 2] << 8) + data_bytes[offset + taxel_idx * 3 * 2 + 3])
                z = ~ ((data_bytes[offset + taxel_idx * 3 * 2 + 4] << 8) + data_bytes[offset + taxel_idx * 3 * 2 + 5])
                sample_raw[sensor_idx, taxel_idx, :] = np.array([x, y, z])

                if not self.reading_first_sample:
                    # Check for large jumps (overflow) and correct them
                    # code below assumes only one overflow threshold is crossed
                    if abs(x - self.prev_x[sensor_idx][taxel_idx]) >= 2 ** 11:
                        if not self.cycle_x[sensor_idx][taxel_idx]:
                            self.cycle_x[sensor_idx][taxel_idx] = x - self.prev_x[sensor_idx][taxel_idx]
                        else:
                            self.cycle_x[sensor_idx][taxel_idx] = 0
                        logger.warning(
                            f'sensor{sensor_idx} taxel{taxel_idx} axis x overflow:  x: {x} ; prev_x: {self.prev_x[sensor_idx][taxel_idx]} ; diff: {x - self.prev_x[sensor_idx][taxel_idx]} ; cycle: {self.cycle_x[sensor_idx][taxel_idx]}')
                    if abs(y - self.prev_y[sensor_idx][taxel_idx]) >= 2 ** 11:
                        if not self.cycle_y[sensor_idx][taxel_idx]:
                            self.cycle_y[sensor_idx][taxel_idx] = y - self.prev_y[sensor_idx][taxel_idx]
                        else:
                            self.cycle_y[sensor_idx][taxel_idx] = 0
                        logger.warning(
                            f'sensor{sensor_idx} taxel{taxel_idx} axis y overflow:  y: {y} ; prev_y: {self.prev_y[sensor_idx][taxel_idx]} ; diff: {y - self.prev_y[sensor_idx][taxel_idx]} ; cycle: {self.cycle_y[sensor_idx][taxel_idx]}')
                    if abs(z - self.prev_z[sensor_idx][taxel_idx]) >= 2 ** 11:  # with temp compensation on, the z-overflow is around 2**12 for some reason
                        if not self.cycle_z[sensor_idx][taxel_idx]:
                            self.cycle_z[sensor_idx][taxel_idx] = z - self.prev_z[sensor_idx][taxel_idx]
                        else:
                            self.cycle_z[sensor_idx][taxel_idx] = 0
                        logger.warning(
                            f'sensor{sensor_idx} taxel{taxel_idx} axis z overflow:  z: {z} ; prev_z: {self.prev_z[sensor_idx][taxel_idx]} ; diff: {z - self.prev_z[sensor_idx][taxel_idx]} ; cycle: {self.cycle_z[sensor_idx][taxel_idx]}')
                self.prev_x[sensor_idx][taxel_idx] = x
                self.prev_y[sensor_idx][taxel_idx] = y
                self.prev_z[sensor_idx][taxel_idx] = z
                x -= self.cycle_x[sensor_idx][taxel_idx]
                y -= self.cycle_y[sensor_idx][taxel_idx]
                z -= self.cycle_z[sensor_idx][taxel_idx]
                sample_raw[sensor_idx, taxel_idx, :] = np.array([x, y, z])

                x *= self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][0]
                y *= self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][0]
                z *= self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][1]

                x += self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][0] * 2 ** 16
                y += self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][0] * 2 ** 16
                z += self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][1] * 2 ** 16

                sample[sensor_idx, taxel_idx, :] = np.array([x, y, z])
        return sample, sample_raw

    async def calibrate(self):
        logger.warning("Starting calibration... DO NOT TOUCH SENSOR ARRAY")
        await self.connect()
        asyncio.create_task(self.subscribe(callback=self.calibration_callback))
        while self.calibration_ctr < self.config.WINDOW_SIZE:
            await asyncio.sleep(0.5)
        await self.unsubscribe()
        for sensor_idx in range(self.num_sensors):
            for i in range(self.config.NUM_TAXELS):
                self.means[sensor_idx, i, 0] = np.mean(self.calibration_samples[:, sensor_idx, i, 0])
                self.means[sensor_idx, i, 1] = np.mean(self.calibration_samples[:, sensor_idx, i, 1])
                self.means[sensor_idx, i, 2] = np.mean(self.calibration_samples[:, sensor_idx, i, 2])
        logger.info("Calibration done! You can now touch the sensor.")
        logger.info(f"Mean values are: {self.means}")

    async def calibration_callback(self, handle: int, data: bytearray):
        if self.calibration_ctr >= self.config.WINDOW_SIZE:
            return
        data_bytes = list(data)
        sample, _ = self.data_convert(data_bytes)
        self.calibration_samples[self.calibration_ctr] = sample
        self.calibration_ctr += 1
        logger.debug(f"Gathered {self.calibration_ctr} samples.")
        self.reading_first_sample = False

    async def data_callback(self, handle: int, data: bytearray):
        data_bytes = list(data)
        row = {}
        sample, sample_raw = self.data_convert(data_bytes)
        for sensor_idx in range(self.num_sensors):
            d = {}
            for taxel_idx in range(self.config.NUM_TAXELS):
                # offset data
                sample[sensor_idx, taxel_idx] -= self.means[sensor_idx, taxel_idx]  # / self.config.SCALE_FACTOR
                d[f'X{taxel_idx}'] = [sample[sensor_idx, taxel_idx, 0]]
                d[f'Y{taxel_idx}'] = [sample[sensor_idx, taxel_idx, 1]]
                d[f'Z{taxel_idx}'] = [sample[sensor_idx, taxel_idx, 2]]
            df = pd.DataFrame(data=d)
            if self.taxel_models[sensor_idx]:
                Y, _ = self.taxel_models[sensor_idx].predict(df)
                Y = Y.flatten()
                for taxel_idx in range(self.config.NUM_TAXELS):
                    predicted_forces = Y[taxel_idx * 3: (taxel_idx + 1) * 3]
                    # Clamp values to range seen in calibration
                    predicted_x = min(self.config.MAX_X, max(self.config.MIN_X, predicted_forces[0]))
                    predicted_y = min(self.config.MAX_Y, max(self.config.MIN_Y, predicted_forces[1]))
                    predicted_z = min(self.config.MAX_Z, max(self.config.MIN_Z, predicted_forces[2]))
                    raw_x = sample_raw[sensor_idx, taxel_idx, 0]
                    raw_y = sample_raw[sensor_idx, taxel_idx, 1]
                    raw_z = sample_raw[sensor_idx, taxel_idx, 2]
                    self.taxels[sensor_idx, taxel_idx] = MagTouchTaxel(x=predicted_x, y=predicted_y, z=predicted_z)
                    self.taxels_raw[sensor_idx, taxel_idx] = MagTouchTaxel(x=raw_x, y=raw_y, z=raw_z)
                    row[f'F_x{sensor_idx}{taxel_idx}'] = predicted_forces[0]
                    row[f'F_y{sensor_idx}{taxel_idx}'] = predicted_forces[1]
                    row[f'F_z{sensor_idx}{taxel_idx}'] = predicted_forces[2]
                    row[f'X{sensor_idx}{taxel_idx}'] = raw_x
                    row[f'Y{sensor_idx}{taxel_idx}'] = raw_y
                    row[f'Z{sensor_idx}{taxel_idx}'] = raw_z
                self.taxels[sensor_idx] = self.taxels[sensor_idx][[3, 2, 0, 1]]
                self.magtouch_data[sensor_idx].taxels = self.taxels[sensor_idx]
                self.magtouch_data_raw[sensor_idx].taxels = self.taxels_raw[sensor_idx]
                self.data_publishers[sensor_idx].publish_sensor_data(self.magtouch_data[sensor_idx])
                self.raw_data_publishers[sensor_idx].publish_sensor_data(self.magtouch_data_raw[sensor_idx])
            else:
                for taxel_idx in range(self.config.NUM_TAXELS):
                    raw_x = sample_raw[sensor_idx, taxel_idx, 0]
                    raw_y = sample_raw[sensor_idx, taxel_idx, 1]
                    raw_z = sample_raw[sensor_idx, taxel_idx, 2]
                    self.taxels_raw[sensor_idx, taxel_idx] = MagTouchTaxel(x=raw_x, y=raw_y, z=raw_z)
                    row[f'X{sensor_idx}{taxel_idx}'] = raw_x
                    row[f'Y{sensor_idx}{taxel_idx}'] = raw_y
                    row[f'Z{sensor_idx}{taxel_idx}'] = raw_z
                self.magtouch_data_raw[sensor_idx].taxels = self.taxels_raw[sensor_idx]
                self.raw_data_publishers[sensor_idx].publish_sensor_data(self.magtouch_data_raw[sensor_idx])

        # Send data to WebSocket as JSON
        if self.config.ENABLE_WS:
            await self.websocket.async_send_data_to_websocket(row)
        self.reading_first_sample = False


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    parser = argparse.ArgumentParser(description='Read data from one or more MagTouch sensors over a serial connection.'
                                                 'The data will be published to topics MagTouchX, X=0,1,...')
    args = parser.parse_args()
    magtouch_reader = MagTouchBleReader(config=MagTouchBleReaderConfig(
        ENABLE_WS=True,
        NUM_SENSORS=1,
        MODEL_NAMES=np.array(["2b_stageII"]),
        uuid=SensorUuid.DATA_CHAR_MAGTOUCH,
        device_mac=DeviceMAC.ARDUINO3
    ))
    magtouch_reader.run()
