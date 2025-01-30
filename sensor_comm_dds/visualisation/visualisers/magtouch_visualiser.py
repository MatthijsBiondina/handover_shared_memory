import time

import numpy as np
from sensor_comm_dds.visualisation.view.bubble_view import BubbleView
from sensor_comm_dds.visualisation.viewmodel.magtouch_viewmodel import MagTouchViewModel
from sensor_comm_dds.visualisation.visualisers.visualiser import Visualiser
from sensor_comm_dds.communication.data_classes.magtouch4 import MagTouch4
from cyclonedds.util import duration
from loguru import logger
import os


class MagTouchVisualiser(Visualiser):
    def __init__(self):
        super().__init__(topic_data_type=MagTouch4, description="Visualise data from a MagTouch sensor.")

        self.view = BubbleView(name=self.topic_name, grid_size=(2, 2))
        self.viewmodel = MagTouchViewModel(view=self.view)

    def run(self):
        data = np.zeros((2, 2, 3))
        ctr = 0
        t_start = time.time()
        for sample in self.reader.take_iter(timeout=duration(seconds=10)):
            for i, taxel in enumerate(sample.taxels):
                data[i//2, i%2] = np.array([taxel.y, taxel.x, taxel.z])  # note reordering of x and y from fingertip frame to visualisation frame
            #logger.debug(data)
            self.viewmodel.update_view(data)
            ctr += 1
            if ctr == 100:
                logger.info(f'MagTouch visualiser running at {ctr/(time.time() - t_start)} Hz')
                ctr = 0
                t_start = time.time()


if __name__ == "__main__":
    logger.info(f"Running {os.path.basename(__file__)}")
    visualiser = MagTouchVisualiser()
    visualiser.run()
