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
class MagTouchSerialReaderConfig(SerialConfig, WebsocketConfig, CycloneConfig):
    ARRAY_SIZE: int = 4  # Number of sensors in the array
    MODEL_NAME: str = "remko-2x2-003"  # Name of the model to load for processing
    WINDOW_SIZE: int = 100  # Number of samples to collect before starting to predict
    SCALE_FACTOR: int = 1000  # Scale factor for the input data
    GAIN: int = 4  # Gain setting (same as firmware)
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


class MagTouchReader:
    def __init__(self, config: MagTouchSerialReaderConfig):
        self.config = config
        self.data_publisher = DataPublisher(topic_name=self.config.topic_names[0], topic_data_type=MagTouch4)
        self.websocket = Websocket(websocket_server_url=self.config.websocket_server_url)

        self.taxel_models = dill.load(
            open(os.path.join(python_src_root, f'calibration/magtouch/models/{self.config.MODEL_NAME}'),
                 'rb'))
        self.ser = serial.Serial(self.config.COM, self.config.BAUD)

        self.means = np.zeros((self.config.ARRAY_SIZE, 3))
        self.calibrate()

    def calibrate(self):
        logger.warning("Starting calibration... DO NOT TOUCH SENSOR ARRAY")
        means_tmp = np.zeros((self.config.WINDOW_SIZE, self.config.ARRAY_SIZE, 3))
        for k in range(self.config.WINDOW_SIZE):
            sample, _ = self.read_sample()
            means_tmp[k] = sample
        for i in range(self.config.ARRAY_SIZE):
            self.means[i, 0] = np.mean(means_tmp[:, i, 0])
            self.means[i, 1] = np.mean(means_tmp[:, i, 1])
            self.means[i, 2] = np.mean(means_tmp[:, i, 2])
        logger.info("Calibration done! You can now touch the sensor.")
        logger.info(f"Mean values are: {self.means}")

    def read_sample(self):
        sample = np.zeros((self.config.ARRAY_SIZE, 3))
        sample_raw = np.zeros((self.config.ARRAY_SIZE, 3))
        while True:
            data = self.ser.read(1)
            if data == b'\xAA':
                data_bytes = self.ser.read(6 * self.config.ARRAY_SIZE)
                for i in range(self.config.ARRAY_SIZE):
                    x = (data_bytes[i * 3 * 2] << 8) + data_bytes[i * 3 * 2 + 1]
                    y = (data_bytes[i * 3 * 2 + 2] << 8) + data_bytes[i * 3 * 2 + 3]
                    z = (data_bytes[i * 3 * 2 + 4] << 8) + data_bytes[i * 3 * 2 + 5]
                    # if i == 3:)
                    #    print(f's{i} z MSB {data_bytes[i * 3 * 2 + 4]} ; z LSB {data_bytes[i * 3 * 2 + 5]}')
                    sample_raw[i, :] = np.array([x, y, z])

                    # TODO: it makes no sense that only x and y should be interpreted as 2's complement, investigate
                    x = (interpret_16b_as_twos_complement(x)) * \
                        self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][0]
                    y = (interpret_16b_as_twos_complement(y)) * \
                        self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][0]
                    z = (z - 32768) * self.config.mlx90393_lsb_lookup[0][self.config.GAIN][self.config.RESOLUTION][1]
                    sample[i, :] = np.array([x, y, z])
                return sample, sample_raw

    def run(self):
        taxels = np.array([MagTouchTaxel(x=0, y=0, z=0) for _ in range(self.config.ARRAY_SIZE)])  # preallocate
        magtouch_data = MagTouch4(taxels)
        while True:
            sample, sample_raw = self.read_sample()
            row = {"t": time.time()}
            for i in range(self.config.ARRAY_SIZE):
                # Predict the force
                X = (sample[i] - self.means[i]) / self.config.SCALE_FACTOR
                Y = self.taxel_models[i].predict(X.reshape(1, -1))
                taxels[i] = MagTouchTaxel(x=-Y[0, 1], y=-Y[0, 0], z=Y[0, 2])
                row[f'F_x{i}'] = Y[0, 0]
                row[f'F_y{i}'] = Y[0, 1]
                row[f'F_z{i}'] = Y[0, 2]
                row[f'X{i}'] = sample_raw[i, 0]
                row[f'Y{i}'] = sample_raw[i, 1]
                row[f'Z{i}'] = sample_raw[i, 2]
            taxels = taxels[[3, 2, 0, 1]]
            magtouch_data.taxels = taxels
            self.data_publisher.publish_sensor_data(magtouch_data)
            # Send data to WebSocket as JSON
            if self.config.ENABLE_WS:
                self.websocket.send_data_to_websocket(row)


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    parser = argparse.ArgumentParser(description='Read data from a MagTouch sensor over a serial connection.')
    parser.add_argument('topic_name', type=str, help='Name of the topic where the sensor data is published')
    args = parser.parse_args()
    topic_name = args.topic_name

    magtouch_reader = MagTouchReader(config=MagTouchSerialReaderConfig(ENABLE_WS=False, topic_names=np.array([topic_name])))
    magtouch_reader.run()
