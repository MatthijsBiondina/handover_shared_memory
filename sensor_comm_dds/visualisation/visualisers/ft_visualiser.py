import numpy as np
from loguru import logger
from cyclonedds.util import duration
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from sensor_comm_dds.visualisation.visualisers.visualiser import Visualiser
from sensor_comm_dds.communication.data_classes.sequence import Sequence
from sensor_comm_dds.communication.readers.data_publisher import DataPublisher
import os
import threading
import time


class SampleReaderThread(QtCore.QThread):

    sample_signal = QtCore.pyqtSignal(object)

    def __init__(self, cyclone_reader):
        QtCore.QThread.__init__(self)
        self.cyclone_reader = cyclone_reader
        self.ctr = 0

    def run(self):
        logger.info('Running SampleReaderThread')
        t_start = time.time()
        for sample in self.cyclone_reader.take_iter(timeout=duration(seconds=10)):
            self.sample_signal.emit(sample)
            self.ctr += 1
            if self.ctr == 100:
                logger.info(f'FT readout frequency: {100/(time.time() - t_start)}')
                self.ctr = 0
                t_start = time.time()


class FTVisualiser(Visualiser):
    def __init__(self):
        super().__init__(topic_data_type=Sequence, description="Visualise data from a UR FT sensor.")
        self.samples_per_frame = 200
        self.plotted_values = np.array([[0.0 for __ in range(6)] for _ in range(self.samples_per_frame)])
        #self.timebase = np.arange(self.samples_per_frame) / self.fs
        self.timebase = np.arange(self.samples_per_frame)

        self.plotWidget = pg.plot(title="Force data")
        self.plot_data_item_ch1 = self.plotWidget.plot(pen=1)
        self.plot_data_item_ch2 = self.plotWidget.plot(pen=2)
        self.plot_data_item_ch3 = self.plotWidget.plot(pen=3)
        self.plotItem = self.plotWidget.getPlotItem()
        self.yrange = None
        self.xrange = None
        if self.xrange:
            self.plotItem.setXRange(self.xrange[0], self.xrange[1])
        if self.yrange:
            self.plotItem.setYRange(self.yrange[0], self.yrange[1])

        self.sample_reader_thread = SampleReaderThread(self.reader)
        self.sample_reader_thread.sample_signal.connect(self.update_graph)
        self.sample_reader_thread.start()

        self.frame_publisher = DataPublisher(topic_name="FTFrame", topic_data_type=Sequence)

    def update_graph(self, sample):
        self.plotted_values = np.roll(self.plotted_values, -1, axis=0)  # moves all samples up a row
        self.plotted_values[-1, :] = np.array(sample.values)
        self.plot_data_item_ch1.setData(self.timebase, self.plotted_values[:, 0])
        self.plot_data_item_ch2.setData(self.timebase, self.plotted_values[:, 1])
        self.plot_data_item_ch3.setData(self.timebase, self.plotted_values[:, 2])

        frame_to_publish = Sequence(values=[self.plotted_values.shape[0], self.plotted_values.shape[1]] + list(self.plotted_values.flatten()))  # first two values are shape
        self.frame_publisher.publish_sensor_data(frame_to_publish)


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    app = QApplication([])
    print(f'Main thread is {threading.current_thread().name}')
    visualiser = FTVisualiser()
    app.exec()
