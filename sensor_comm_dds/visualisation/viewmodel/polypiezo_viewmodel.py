import numpy as np
from sensor_comm_dds.visualisation.viz_utils import color_fader_rgb255


class PolyPiezoViewModel:
    def __init__(self, view, c1='#434A52', c2='#6595BF'):
        self.view = view
        self.grid_size = self.view.grid_size
        self.c1 = c1
        self.c2 = c2

    def data_to_formatted_rgb(self, data):
        """
        Converts data from a single sensor to a proper RGB format for the GridPlot
        """
        data_resize = np.array(data).reshape(self.grid_size)
        data_rgb = [[color_fader_rgb255(self.c1, self.c2, mix=data_resize[i, j]/128) for i in range(self.grid_size[0])]
                    for j in range(self.grid_size[1])]
        return data_rgb

    def update_view(self, data):
        self.view.square_colors = self.data_to_formatted_rgb(data)
        self.view.text = {(column, row): str(data[row * self.grid_size[1] + column]) for row in range(self.grid_size[0])
                          for column in range(self.grid_size[1])}
        self.view.update_view()
