import os
import csv
from loguru import logger
from datetime import *

import argparse
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader

from sensor_comm_dds.communication.data_classes.envpressure import EnvironmentPressure
from sensor_comm_dds.communication.data_classes.irtouch32 import IRTouch32
from sensor_comm_dds.communication.data_classes.magtouch4 import MagTouch4
from sensor_comm_dds.utils.liveliness_listener import LivelinessListener


def init_directory(directory):
    """
    Checks if directory exists, if not, creates it
    :param directory: directory to check/create
    """
    logger.info(f'directory is {directory}')
    if not os.path.exists(directory):
        ans = input("Make new directory (" + directory + ")? [Y/N] ")
        if ans.lower() == "y":
            os.makedirs(directory)
        elif ans.lower() == "n":
            raise NotADirectoryError("Data directory doesn't exist, creation cancelled by user.")
        else:
            raise ValueError("Invalid answer given, enter [Y] or [N].")

def _create_unique_file_name(directory, preamble=None, extension='.csv'):
    # Build the file name so that each experiment (a.k.a. each run of the code) saves data to a different file

    file_name = str(date.today())
    if preamble:
        file_name = preamble + "_" + file_name
    file_number = 0
    for file_in_dir in os.listdir(directory):
        if file_in_dir.startswith(file_name):
            val = int(file_in_dir[file_in_dir.find("[") + 1:file_in_dir.find("]")])
            if val > file_number:
                file_number = val
    file_name = file_name + "[" + str(file_number + 1) + "]" + extension

    return file_name

class DataLogger:
    def __init__(self, directory, buffer_size=10, file_preamble=None):
        self.directory = directory
        init_directory(self.directory)
        self.file_name = self.create_unique_file_name(preamble=file_preamble)
        self.csv_header = None
        self.buffer = []
        self.max_buffer_size = buffer_size
        self.current_data = None

    def create_unique_file_name(self, preamble=None, extension='.csv'):
        return _create_unique_file_name(directory=self.directory)

    def init_csv_file(self, directory, file_name):
        if self.csv_header:
            with open(os.path.join(directory, file_name), 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=';')
                csv_writer.writerow(self.csv_header)

    def persist(self, data):
        """
        Persists a single data sample
        :param data: list, the data
        """
        self.current_data = data
        self.buffer.append([datetime.now()] + data)

        if len(self.buffer) >= self.max_buffer_size:
            self._persist_to_file()
            self.buffer = []

    def _persist_to_file(self):
        """
        Persists the complete buffer to a .csv file
        """
        with open(os.path.join(self.directory, self.file_name), 'a+', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';')
            for row in self.buffer:
                csv_writer.writerow(row)
        logger.debug(f"Wrote to file {os.path.join(self.directory, self.file_name)}")


class CycloneDataLogger(DataLogger):
    def __init__(self, topic_name, topic_data_type, directory, buffer_size=10, file_preamble=None):
        super().__init__(directory=directory, buffer_size=buffer_size, file_preamble=topic_name)
        self.topic_name = topic_name
        listener = LivelinessListener(topic_name=topic_name)
        domain_participant = DomainParticipant()
        topic = Topic(domain_participant, self.topic_name, topic_data_type)
        self.reader = DataReader(domain_participant, topic, listener=listener)  # you can also supply Subscriber
        # instead of DomainParticipant to the DataReader constructor, but is not necessary

    def unpack_sample(self, sample):
        raise NotImplementedError

    def read_single_sample(self):
        return self.unpack_sample(self.reader.read(N=1)[0])

    def run(self):
        for sample in self.reader.take_iter(timeout=None):
            self.persist(self.unpack_sample(sample))


class IRTouchDataLogger(CycloneDataLogger):
    def __init__(self, topic_name, directory, buffer_size=10, grid_size=(5, 4, 5, 4, 5, 4, 5)):
        super().__init__(topic_name=topic_name, topic_data_type=IRTouch32, directory=directory, buffer_size=buffer_size)
        num_taxels = sum(list(grid_size))
        self.csv_header = ['timestamp', 'strain'] + [f'taxel{i}' for i in range(num_taxels)]
        self.init_csv_file(self.directory, self.file_name)

    def unpack_sample(self, sample: IRTouch32):
        return [sample.strain_value] + list(sample.taxel_values)


class MagTouchDataLogger(CycloneDataLogger):
    def __init__(self, topic_name, directory="./data/smart_textile", buffer_size=10, grid_size=(2, 2)):
        super().__init__(topic_name=topic_name, topic_data_type=MagTouch4, directory=directory, buffer_size=buffer_size)
        grid_height = grid_size[0]
        grid_width = grid_size[1]
        self.num_taxels = grid_width * grid_height
        self.csv_header = ['PCB addr', 'timestamp']
        for i in range(self.num_taxels):
            self.csv_header += [f'taxel{i}_x', f'taxel{i}_y', f'taxel{i}_z']
        self.init_csv_file(self.directory, self.file_name)

    def unpack_sample(self, sample: MagTouch4):
        values = [0 for _ in range(self.num_taxels * 3)]
        for i, taxel in enumerate(sample.taxels):
            values[i * 3] = taxel.x
            values[i * 3 + 1] = taxel.y
            values[i * 3 + 2] = taxel.z
        return values


class BMP384DataLogger(CycloneDataLogger):
    def __init__(self, topic_name, directory, buffer_size=10):
        super().__init__(topic_name=topic_name, topic_data_type=EnvironmentPressure, directory=directory, buffer_size=buffer_size)
        self.csv_header = ['timestamp', 'pressure', 'temperature']
        self.init_csv_file(self.directory, self.file_name)

    def unpack_sample(self, sample: EnvironmentPressure):
        return [sample.pressure, sample.temperature]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This reader will log the data received on a topic to a .csv file.")
    parser.add_argument('topic_name', type=str, help='Name of the topic where the sensor data is published')
    parser.add_argument('sensor_type', type=str, help='Sensor type that is publishing to the topic: '
                                                      'IRTOUCH32, MAGTOUCH, BMP384')
    args = parser.parse_args()
    topic_name = args.topic_name
    data_root = '/home/idlab403/data/'  # TODO: parse from config
    if args.sensor_type == "IRTOUCH32":
        data_handler = IRTouchDataLogger(topic_name=topic_name, directory=os.path.join(data_root, 'irtouch32/'))
    elif args.sensor_type == "MAGTOUCH":
        data_handler = MagTouchDataLogger(topic_name=topic_name, directory=os.path.join(data_root, 'magtouch/'))
    elif args.sensor_type == "BMP384":
        data_handler = BMP384DataLogger(topic_name=topic_name, directory=os.path.join(data_root, 'bmp384/'))
    else:
        raise ValueError("Invalid sensor type passed, check the help [-h] for a list of options")
    data_handler.run()
