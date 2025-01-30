from sensor_comm_dds.communication.data_classes.sequence import Sequence
from sensor_comm_dds.visualisation.view.square_grid_view import SquareGridView
from sensor_comm_dds.visualisation.viewmodel.switches_viewmodel import SwitchesViewModel
from sensor_comm_dds.visualisation.visualisers.visualiser import Visualiser
from cyclonedds.util import duration
import os
from loguru import logger


class SwitchesVisualiser(Visualiser):
    def __init__(self, grid_size=(1, 2)):
        super().__init__(topic_data_type=Sequence, description="Visualise data from Switches.")

        self.view = SquareGridView(name=self.topic_name, grid_size=(grid_size), disp_vals=False, cell_width=150)
        self.viewmodel = SwitchesViewModel(view=self.view)

    def unpack_data_sample(self, sample):
        return sample.values

    def run(self):
        for sample in self.reader.take_iter(timeout=duration(seconds=10)):
            taxel_values = self.unpack_data_sample(sample)
            self.viewmodel.update_view(taxel_values)


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    visualiser = SwitchesVisualiser()
    visualiser.run()
