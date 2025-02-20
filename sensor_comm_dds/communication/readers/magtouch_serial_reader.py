import argparse
import dill
import numpy as np
import os
import pandas as pd
import serial
import time

from cantrips.logging.logger import get_logger
from cyclone.patterns.ddswriter import DDSWriter

logger = get_logger()
# from loguru import logger

from cyclone.cyclone_participant import CycloneParticipant
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
    NUM_TAXELS: int = 4  # Number of sensors in the array
    MODEL_NAMES: np.ndarray = np.array(
        ["remko-2x2-003", "remko-2x2-003"]
    )  # Names of the model to load for processing
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
            [[0.150, 0.242], [0.300, 0.484], [0.601, 0.968], [1.202, 1.936]],
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
            [[0.157, 0.253], [0.315, 0.507], [0.629, 1.014], [1.258, 2.027]],
        ],
    ]


class MagTouchSerialReader:
    def __init__(self, config: MagTouchSerialReaderConfig):
        self.participant = CycloneParticipant()
        self.config = config
        self.data_publishers = []
        self.raw_data_publishers = []
        for i in range(self.config.NUM_SENSORS):
            self.data_publishers.append(
                DDSWriter(
                    self.participant,
                    topic_name="MagTouch" + str(i),
                    idl_dataclass=MagTouch4,
                )
            )
            self.raw_data_publishers.append(
                DDSWriter(
                    self.participant,
                    topic_name="MagTouchRaw" + str(i),
                    idl_dataclass=MagTouch4,
                )
            )

            # self.data_publishers.append(DataPublisher(topic_name="MagTouch" + str(i), topic_data_type=MagTouch4))
            # self.raw_data_publishers.append(DataPublisher(topic_name="MagTouchRaw" + str(i), topic_data_type=MagTouch4))
        self.websocket = Websocket(
            websocket_server_url=self.config.websocket_server_url
        )

        self.taxel_models = [None for _ in range(self.config.NUM_SENSORS)]
        for sensor_idx in range(self.config.NUM_SENSORS):
            if self.config.MODEL_NAMES[sensor_idx]:
                self.taxel_models[sensor_idx] = dill.load(
                    open(
                        os.path.join(
                            python_src_root,
                            f"calibration/magtouch/models/{self.config.MODEL_NAMES[sensor_idx]}",
                        ),
                        "rb",
                    )
                )
        self.ser = serial.Serial(self.config.COM, self.config.BAUD)

        # TODO: update for multiple fingertips
        self.prev_x = [0] * self.config.NUM_TAXELS
        self.prev_y = [0] * self.config.NUM_TAXELS
        self.prev_z = [0] * self.config.NUM_TAXELS
        self.cycle_x = [0] * self.config.NUM_TAXELS
        self.cycle_y = [0] * self.config.NUM_TAXELS
        self.cycle_z = [0] * self.config.NUM_TAXELS
        self.means = np.zeros((self.config.NUM_SENSORS, self.config.NUM_TAXELS, 3))
        self.calibrate()

    def calibrate(self):
        logger.warning("Starting calibration... DO NOT TOUCH SENSOR ARRAY")
        self.init_prev_vals()
        means_tmp = np.zeros(
            (
                self.config.WINDOW_SIZE,
                self.config.NUM_SENSORS,
                self.config.NUM_TAXELS,
                3,
            )
        )
        for k in range(self.config.WINDOW_SIZE):
            sample, _ = self.read_sample()
            means_tmp[k] = sample
        for sensor_idx in range(self.config.NUM_SENSORS):
            for i in range(self.config.NUM_TAXELS):
                self.means[sensor_idx, i, 0] = np.mean(means_tmp[:, sensor_idx, i, 0])
                self.means[sensor_idx, i, 1] = np.mean(means_tmp[:, sensor_idx, i, 1])
                self.means[sensor_idx, i, 2] = np.mean(means_tmp[:, sensor_idx, i, 2])
        logger.info("Calibration done! You can now touch the sensor.")
        # logger.info(f"Mean values are: {self.means}")

    def init_prev_vals(self):
        while self.ser.read(1) != b"\xAA":
            pass
        data_bytes = self.ser.read(6 * self.config.NUM_TAXELS)
        for i in range(self.config.NUM_TAXELS):
            self.prev_x[i] = ~((data_bytes[i * 3 * 2] << 8) + data_bytes[i * 3 * 2 + 1])
            self.prev_y[i] = ~(
                (data_bytes[i * 3 * 2 + 2] << 8) + data_bytes[i * 3 * 2 + 3]
            )
            self.prev_z[i] = ~(
                (data_bytes[i * 3 * 2 + 4] << 8) + data_bytes[i * 3 * 2 + 5]
            )

    def read_sample(self):
        sample = np.zeros((self.config.NUM_SENSORS, self.config.NUM_TAXELS, 3))
        sample_raw = np.zeros((self.config.NUM_SENSORS, self.config.NUM_TAXELS, 3))
        while True:
            data = self.ser.read(1)
            if data == b"\xAA":
                data_bytes = self.ser.read(
                    self.config.NUM_SENSORS * 6 * self.config.NUM_TAXELS
                )
                for sensor_idx in range(self.config.NUM_SENSORS):
                    for taxel_idx in range(self.config.NUM_TAXELS):
                        offset = sensor_idx * 6 * self.config.NUM_TAXELS
                        x = ~(
                            (data_bytes[offset + taxel_idx * 3 * 2] << 8)
                            + data_bytes[offset + taxel_idx * 3 * 2 + 1]
                        )
                        y = ~(
                            (data_bytes[offset + taxel_idx * 3 * 2 + 2] << 8)
                            + data_bytes[offset + taxel_idx * 3 * 2 + 3]
                        )
                        z = ~(
                            (data_bytes[offset + taxel_idx * 3 * 2 + 4] << 8)
                            + data_bytes[offset + taxel_idx * 3 * 2 + 5]
                        )

                        # Check for large jumps (overflow) and correct them
                        # code below assumes only one overflow threshold is crossed
                        if abs(x - self.prev_x[taxel_idx]) >= 2**11:
                            if not self.cycle_x[taxel_idx]:
                                self.cycle_x[taxel_idx] = x - self.prev_x[taxel_idx]
                            else:
                                self.cycle_x[taxel_idx] = 0
                            logger.warning(
                                f"sensor {sensor_idx} taxel {taxel_idx} axis x overflow diff: {x - self.prev_x[taxel_idx]} ; x: {x} ; prev_x: {self.prev_x[taxel_idx]}"
                            )
                        if abs(y - self.prev_y[taxel_idx]) >= 2**11:
                            if not self.cycle_y[taxel_idx]:
                                self.cycle_y[taxel_idx] = y - self.prev_y[taxel_idx]
                            else:
                                self.cycle_y[taxel_idx] = 0
                            logger.warning(
                                f"sensor {sensor_idx} taxel {taxel_idx} axis y overflow diff: {y - self.prev_y[taxel_idx]} ; y: {y} ; prev_y: {self.prev_y[taxel_idx]}"
                            )
                        if (
                            abs(z - self.prev_z[taxel_idx]) >= 2**11
                        ):  # with temp compensation on, the z-overflow is around 2**12 for some reason
                            if not self.cycle_z[taxel_idx]:
                                self.cycle_z[taxel_idx] = z - self.prev_z[taxel_idx]
                            else:
                                self.cycle_z[taxel_idx] = 0
                            logger.warning(
                                f"sensor {sensor_idx} taxel {taxel_idx} axis z overflow diff: {z - self.prev_z[taxel_idx]} ; z: {z} ; prev_z: {self.prev_z[taxel_idx]}"
                            )
                        self.prev_x[taxel_idx] = x
                        self.prev_y[taxel_idx] = y
                        self.prev_z[taxel_idx] = z
                        x -= self.cycle_x[taxel_idx]
                        y -= self.cycle_y[taxel_idx]
                        z -= self.cycle_z[taxel_idx]
                        sample_raw[sensor_idx, taxel_idx, :] = np.array([x, y, z])

                        x *= self.config.mlx90393_lsb_lookup[0][self.config.GAIN][
                            self.config.RESOLUTION
                        ][0]
                        y *= self.config.mlx90393_lsb_lookup[0][self.config.GAIN][
                            self.config.RESOLUTION
                        ][0]
                        z *= self.config.mlx90393_lsb_lookup[0][self.config.GAIN][
                            self.config.RESOLUTION
                        ][1]

                        x += (
                            self.config.mlx90393_lsb_lookup[0][self.config.GAIN][
                                self.config.RESOLUTION
                            ][0]
                            * 2**16
                        )
                        y += (
                            self.config.mlx90393_lsb_lookup[0][self.config.GAIN][
                                self.config.RESOLUTION
                            ][0]
                            * 2**16
                        )
                        z += (
                            self.config.mlx90393_lsb_lookup[0][self.config.GAIN][
                                self.config.RESOLUTION
                            ][1]
                            * 2**16
                        )

                        # z inversion
                        # z = 1/z

                        sample[sensor_idx, taxel_idx, :] = np.array([x, y, z])
                return sample, sample_raw

    def run(self):
        taxels = np.array(
            [
                [MagTouchTaxel(x=0, y=0, z=0) for _ in range(self.config.NUM_TAXELS)]
                for __ in range(self.config.NUM_SENSORS)
            ]
        )  # preallocate
        taxels_raw = np.array(
            [
                [MagTouchTaxel(x=0, y=0, z=0) for _ in range(self.config.NUM_TAXELS)]
                for __ in range(self.config.NUM_SENSORS)
            ]
        )  # preallocate
        magtouch_data = [MagTouch4(taxels[i]) for i in range(self.config.NUM_SENSORS)]
        magtouch_raw_data = [
            MagTouch4(taxels_raw[i]) for i in range(self.config.NUM_SENSORS)
        ]
        self.init_prev_vals()
        while True:
            sample, sample_raw = self.read_sample()
            row = {"t": time.time()}
            for sensor_idx in range(self.config.NUM_SENSORS):
                d = {}
                for taxel_idx in range(self.config.NUM_TAXELS):
                    # offset data
                    sample[sensor_idx, taxel_idx] -= self.means[
                        sensor_idx, taxel_idx
                    ]  # / self.config.SCALE_FACTOR
                    d[f"X{taxel_idx}"] = [sample[sensor_idx, taxel_idx, 0]]
                    d[f"Y{taxel_idx}"] = [sample[sensor_idx, taxel_idx, 1]]
                    d[f"Z{taxel_idx}"] = [sample[sensor_idx, taxel_idx, 2]]
                df = pd.DataFrame(data=d)
                if self.taxel_models[sensor_idx]:
                    Y, _ = self.taxel_models[sensor_idx].predict(df)
                    Y = Y.flatten()
                    for taxel_idx in range(self.config.NUM_TAXELS):
                        predicted_forces = Y[taxel_idx * 3 : (taxel_idx + 1) * 3]
                        # Clamp values to range seen in calibration
                        predicted_x = min(
                            self.config.MAX_X,
                            max(self.config.MIN_X, predicted_forces[0]),
                        )
                        predicted_y = min(
                            self.config.MAX_Y,
                            max(self.config.MIN_Y, predicted_forces[1]),
                        )
                        predicted_z = min(
                            self.config.MAX_Z,
                            max(self.config.MIN_Z, predicted_forces[2]),
                        )
                        raw_x = sample_raw[sensor_idx, taxel_idx, 0]
                        raw_y = sample_raw[sensor_idx, taxel_idx, 1]
                        raw_z = sample_raw[sensor_idx, taxel_idx, 2]
                        taxels[sensor_idx, taxel_idx] = MagTouchTaxel(
                            x=predicted_x, y=predicted_y, z=predicted_z
                        )
                        taxels_raw[sensor_idx, taxel_idx] = MagTouchTaxel(
                            x=raw_x, y=raw_y, z=raw_z
                        )
                        row[f"F_x{sensor_idx}{taxel_idx}"] = predicted_forces[0]
                        row[f"F_y{sensor_idx}{taxel_idx}"] = predicted_forces[1]
                        row[f"F_z{sensor_idx}{taxel_idx}"] = predicted_forces[2]
                        row[f"X{sensor_idx}{taxel_idx}"] = raw_x
                        row[f"Y{sensor_idx}{taxel_idx}"] = raw_y
                        row[f"Z{sensor_idx}{taxel_idx}"] = raw_z
                    # reshaping from PCB numbering scheme to visualisation numbering scheme
                    taxels[sensor_idx] = taxels[sensor_idx][[3, 2, 0, 1]]

                    magtouch_data[sensor_idx].taxels = taxels[sensor_idx]
                    magtouch_raw_data[sensor_idx].taxels = taxels_raw[sensor_idx]
                    self.data_publishers[sensor_idx](magtouch_data[sensor_idx])
                    self.raw_data_publishers[sensor_idx](magtouch_raw_data[sensor_idx])
                else:
                    for taxel_idx in range(self.config.NUM_TAXELS):
                        raw_x = sample_raw[sensor_idx, taxel_idx, 0]
                        raw_y = sample_raw[sensor_idx, taxel_idx, 1]
                        raw_z = sample_raw[sensor_idx, taxel_idx, 2]
                        taxels_raw[sensor_idx, taxel_idx] = MagTouchTaxel(
                            x=raw_x, y=raw_y, z=raw_z
                        )
                        row[f"X{sensor_idx}{taxel_idx}"] = raw_x
                        row[f"Y{sensor_idx}{taxel_idx}"] = raw_y
                        row[f"Z{sensor_idx}{taxel_idx}"] = raw_z
                    magtouch_raw_data[sensor_idx].taxels = taxels_raw[sensor_idx]

                    self.raw_data_publishers[sensor_idx](magtouch_raw_data[sensor_idx])

                    # self.raw_data_publishers[sensor_idx].publish_sensor_data(
                    #     magtouch_raw_data[sensor_idx]
                    # )

            if self.config.ENABLE_WS:
                self.websocket.send_data_to_websocket(row)


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    parser = argparse.ArgumentParser(
        description="Read data from one or more MagTouch sensors over a serial connection."
        "The data will be published to topics MagTouchX, X=0,1,..."
    )
    args = parser.parse_args()
    magtouch_reader = MagTouchSerialReader(
        config=MagTouchSerialReaderConfig(
            ENABLE_WS=False,
            NUM_SENSORS=1,
            MODEL_NAMES=np.array([None]),
            COM="/dev/ttyACM0",
        )
    )
    magtouch_reader.run()
