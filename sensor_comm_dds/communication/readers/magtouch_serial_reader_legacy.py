import argparse
import dill
import numpy as np
import os
import serial
import time
from loguru import logger

from sensor_comm_dds.communication.config.cyclone_config import CycloneConfig
from sensor_comm_dds.communication.readers.websocket import Websocket
from sensor_comm_dds.utils.paths import python_src_root
from sensor_comm_dds.utils.conversions import interpret_16b_as_twos_complement
from sensor_comm_dds.communication.data_classes.magtouch4 import MagTouch4
from sensor_comm_dds.communication.data_classes.magtouch_taxel import MagTouchTaxel
from sensor_comm_dds.communication.config.serial_config import SerialConfig
from sensor_comm_dds.communication.config.websocket_config import WebsocketConfig
from dataclasses import dataclass
from sensor_comm_dds.communication.readers.data_publisher import DataPublisher


@dataclass
class MagTouchSerialReaderConfig(SerialConfig, WebsocketConfig):
    NUM_SENSORS: int = 1
    ARRAY_SIZE: int = 4  # Number of sensors in the array
    MODEL_NAMES: np.ndarray = np.array(["remko-2x2-003", "remko-2x2-003"])  # Names of the model to load for processing
    WINDOW_SIZE: int = 100  # Number of samples to collect before starting to predict
    SCALE_FACTOR: int = 1000  # Scale factor for the input data
    GAIN: int = 7  # Gain setting (same as firmware)
    RESOLUTION: int = 0  # Resolution setting (same as firmware)
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


class MagTouchSerialReader:
    def __init__(self, config: MagTouchSerialReaderConfig):
        self.config = config
        self.num_sensors = self.config.NUM_SENSORS
        self.data_publishers = []
        for i in range(self.config.NUM_SENSORS):
            self.data_publishers.append(DataPublisher(topic_name="MagTouch" + str(i), topic_data_type=MagTouch4))
        self.websocket = Websocket(websocket_server_url=self.config.websocket_server_url)

        self.taxel_models = []
        for i in range(self.config.NUM_SENSORS):
            self.taxel_models.append(dill.load(
                open(os.path.join(python_src_root, f'calibration/magtouch/models/{self.config.MODEL_NAMES[i]}'),
                     'rb')))
        self.ser = serial.Serial(self.config.COM, self.config.BAUD)

        # TODO: update for multiple fingertips
        self.reading_first_sample = True
        self.prev_x = [0] * self.config.ARRAY_SIZE
        self.prev_y = [0] * self.config.ARRAY_SIZE
        self.prev_z = [0] * self.config.ARRAY_SIZE
        self.cycle_x = [0] * self.config.ARRAY_SIZE
        self.cycle_y = [0] * self.config.ARRAY_SIZE
        self.cycle_z = [0] * self.config.ARRAY_SIZE
        self.means = np.zeros((self.num_sensors, self.config.ARRAY_SIZE, 3))
        self.calibrate()

    def calibrate(self):
        logger.warning("Starting calibration... DO NOT TOUCH SENSOR ARRAY")
        means_tmp = np.zeros((self.config.WINDOW_SIZE, self.num_sensors, self.config.ARRAY_SIZE, 3))
        for k in range(self.config.WINDOW_SIZE):
            sample, _ = self.read_sample()
            means_tmp[k] = sample
        for sensor_idx in range(self.num_sensors):
            for i in range(self.config.ARRAY_SIZE):
                    self.means[sensor_idx, i, 0] = np.mean(means_tmp[:, sensor_idx, i, 0])
                    self.means[sensor_idx, i, 1] = np.mean(means_tmp[:, sensor_idx, i, 1])
                    self.means[sensor_idx, i, 2] = np.mean(means_tmp[:, sensor_idx, i, 2])
        logger.info("Calibration done! You can now touch the sensor.")
        logger.info(f"Mean values are: {self.means}")

    def read_sample(self):
        sample = np.zeros((self.num_sensors, self.config.ARRAY_SIZE, 3))
        sample_raw = np.zeros((self.num_sensors, self.config.ARRAY_SIZE, 3))
        while True:
            data = self.ser.read(1)
            if data == b'\xAA':
                data_bytes = self.ser.read(self.num_sensors * 6 * self.config.ARRAY_SIZE)
                for sensor_idx in range(self.num_sensors):
                    for i in range(self.config.ARRAY_SIZE):
                        offset = sensor_idx * 6 * self.config.ARRAY_SIZE
                        x = ~ ((data_bytes[offset + i * 3 * 2] << 8) + data_bytes[offset + i * 3 * 2 + 1])
                        #x_ = ~ x
                        y = ~ ((data_bytes[offset + i * 3 * 2 + 2] << 8) + data_bytes[offset + i * 3 * 2 + 3])
                        z = ~ ((data_bytes[offset + i * 3 * 2 + 4] << 8) + data_bytes[offset + i * 3 * 2 + 5])
                        # if i == 3:)
                        #    print(f's{i} z MSB {data_bytes[i * 3 * 2 + 4]} ; z LSB {data_bytes[i * 3 * 2 + 5]}')
                        sample_raw[sensor_idx, i, :] = np.array([x, y, z])

                        if not self.reading_first_sample:
                            # Check for overflow and correct
                            if abs(x - self.prev_x[i]) > 2 ** 15:
                                self.cycle_x[i] += 1 if x < self.prev_x[i] else -1
                                logger.warning(f"finger{sensor_idx} taxel{i} x overflow detected")
                            if abs(y - self.prev_y[i]) > 2 ** 15:
                                self.cycle_y[i] += 1 if y < self.prev_y[i] else -1
                                logger.warning(f"finger{sensor_idx} taxel{i} y overflow detected")
                            if abs(z - self.prev_z[i]) > 2 ** 15:
                                self.cycle_z[i] += 1 if z < self.prev_z[i] else -1
                                logger.warning(f"finger{sensor_idx} taxel{i} z overflow detected")
                        self.prev_x[i] = x
                        self.prev_y[i] = y
                        self.prev_z[i] = z
                        x += self.cycle_x[i] * 2 ** 16
                        #x_ += self.cycle_x[i] * 2 ** 16
                        y += self.cycle_y[i] * 2 ** 16
                        z += self.cycle_z[i] * 2 ** 16
                        #sample_raw[sensor_idx, i, :] = np.array([x, y, z])
                        '''# TODO: it makes no sense that only x and y should be interpreted as 2's complement, investigate
                        x = (interpret_16b_as_twos_complement(x)) * \
                            self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][0]
                        y = (interpret_16b_as_twos_complement(y)) * \
                            self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][0]
                        z = (z - 32768) * self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][1]'''

                        x *= self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][0]
                        #x_ *= self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][0]
                        y *= self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][0]
                        z *= self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][1]

                        x += self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][0] * 2 ** 16
                        #x_ += self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][0] * 2 ** 16
                        y += self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][0] * 2 ** 16
                        z += self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][1] * 2 ** 16
                        #logger.debug(f'{x:.2f}\t{x_:.2f}')
                        sample[sensor_idx, i, :] = np.array([x, y, z])
                self.reading_first_sample = False
                #logger.debug(sample)
                return sample, sample_raw

    def run(self):
        taxels = np.array([[MagTouchTaxel(x=0, y=0, z=0) for _ in range(self.config.ARRAY_SIZE)] for __ in range(self.num_sensors)])  # preallocate
        magtouch_data = [MagTouch4(taxels[i]) for i in range(self.num_sensors)]
        while True:
            sample, sample_raw = self.read_sample()
            row = {"t": time.time()}
            for sensor_idx in range(self.num_sensors):
                for i in range(self.config.ARRAY_SIZE):
                    # Predict the force
                    X = (sample[sensor_idx, i] - self.means[sensor_idx, i]) / self.config.SCALE_FACTOR
                    # X = sample[sensor_idx, i] / self.config.SCALE_FACTOR
                    Y = self.taxel_models[sensor_idx][i].predict(X.reshape(1, -1))
                    # logger.debug(f'{i}: {X} -> {Y}')
                    taxels[sensor_idx, i] = MagTouchTaxel(x=-Y[0, 0], y=-Y[0, 1], z=Y[0, 2])
                    row[f'F_x{sensor_idx}{i}'] = Y[0, 0]
                    row[f'F_y{sensor_idx}{i}'] = Y[0, 1]
                    row[f'F_z{sensor_idx}{i}'] = Y[0, 2]
                    row[f'X{sensor_idx}{i}'] = sample_raw[sensor_idx, i, 0]
                    row[f'Y{sensor_idx}{i}'] = sample_raw[sensor_idx, i, 1]
                    row[f'Z{sensor_idx}{i}'] = sample_raw[sensor_idx, i, 2]
                taxels[sensor_idx] = taxels[sensor_idx][[3, 2, 0, 1]]
                magtouch_data[sensor_idx].taxels = taxels[sensor_idx]

                self.data_publishers[sensor_idx].publish_sensor_data(magtouch_data[sensor_idx])
            # Send data to WebSocket as JSON
            if self.config.ENABLE_WS:
                self.websocket.send_data_to_websocket(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read data from one or more MagTouch sensors over a serial connection.'
                                                 'The data will be published to topics MagTouchX, X=0,1,...')
    args = parser.parse_args()
    magtouch_reader = MagTouchSerialReader(config=MagTouchSerialReaderConfig(
        ENABLE_WS=False,
        NUM_SENSORS=1,
        MODEL_NAMES=np.array(["remko-2x2-5-007"]),
        COM='/dev/ttyACM0'
    ))
    magtouch_reader.run()
