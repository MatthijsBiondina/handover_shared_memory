from sensor_comm_dds.communication.data_classes.clotheshanger import Clotheshanger
from sensor_comm_dds.visualisation.view.square_grid_view import SquareGridView
from sensor_comm_dds.visualisation.viewmodel.clotheshanger_viewmodel import ClotheshangerViewModel
from sensor_comm_dds.visualisation.visualisers.visualiser import Visualiser
from cyclonedds.util import duration
from loguru import logger
import os


class ClotheshangerVisualiser(Visualiser):
    def __init__(self):
        super().__init__(topic_data_type=Clotheshanger, description="Visualise data from a Clotheshanger sensor.")

        self.view = SquareGridView(name=self.topic_name, grid_size=(1, 4), disp_vals=True)
        self.viewmodel = ClotheshangerViewModel(view=self.view)

    def unpack_data_sample(self, sample):
        return sample.taxel_values

    def run(self):
        for sample in self.reader.take_iter(timeout=duration(seconds=10)):
            taxel_values = self.unpack_data_sample(sample)
            self.viewmodel.update_view(taxel_values)


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    visualiser = ClotheshangerVisualiser()
    visualiser.run()
