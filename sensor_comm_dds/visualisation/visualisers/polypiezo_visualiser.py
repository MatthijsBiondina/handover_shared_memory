from sensor_comm_dds.communication.data_classes.polypiezo import PolyPiezo
from sensor_comm_dds.visualisation.view.square_grid_view import SquareGridView
from sensor_comm_dds.visualisation.viewmodel.polypiezo_viewmodel import PolyPiezoViewModel
from sensor_comm_dds.visualisation.visualisers.visualiser import Visualiser
from cyclonedds.util import duration
from loguru import logger
import os


class PolyPiezoVisualiser(Visualiser):
    def __init__(self, grid_size=(5, 5)):
        super().__init__(topic_data_type=PolyPiezo, description="Visualise data from a PolyPiezo sensor.")

        self.view = SquareGridView(name=self.topic_name, grid_size=grid_size, disp_vals=True)
        self.viewmodel = PolyPiezoViewModel(view=self.view)

    def unpack_data_sample(self, sample):
        return sample.data_values

    def run(self):
        for sample in self.reader.take_iter(timeout=duration(seconds=10)):
            taxel_values = self.unpack_data_sample(sample)
            self.viewmodel.update_view(taxel_values)


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    visualiser = PolyPiezoVisualiser()
    visualiser.run()
