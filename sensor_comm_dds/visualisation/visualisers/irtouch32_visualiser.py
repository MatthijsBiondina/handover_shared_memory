from sensor_comm_dds.visualisation.view.irtouch32_view import IRTouch32View
from sensor_comm_dds.visualisation.viewmodel.irtouch32_viewmodel import IRTouch32ViewModel
from sensor_comm_dds.visualisation.visualisers.visualiser import Visualiser
from sensor_comm_dds.communication.data_classes.irtouch32 import IRTouch32
from cyclonedds.util import duration
from loguru import logger
import os


class IRTouch32Visualiser(Visualiser):
    def __init__(self):
        super().__init__(topic_data_type=IRTouch32, description="Visualise data from a IRTouch32 sensor.")

        self.view = IRTouch32View(name=self.topic_name)
        self.viewmodel = IRTouch32ViewModel(view=self.view)

    def unpack_data_sample(self, sample):
        return sample.taxel_values, sample.strain_value

    def run(self):
        for sample in self.reader.take_iter(timeout=duration(seconds=10)):
            taxel_values, _ = self.unpack_data_sample(sample)  # TODO: add viz for strain reading
            self.viewmodel.update(taxel_values)


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    visualiser = IRTouch32Visualiser()
    visualiser.run()
