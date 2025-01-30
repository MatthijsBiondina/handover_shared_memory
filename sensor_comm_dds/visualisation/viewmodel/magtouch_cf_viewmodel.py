import numpy as np
from sensor_comm_dds.visualisation.viz_utils import color_fader_rgb255


class MagTouchCFViewModel:
    def __init__(self, view, c1='#434A52', c2='#7DB5A8'):
        self.view = view
        self.grid_size = self.view.grid_size
        self.c1 = c1
        self.c2 = c2
        self.max_xy = 3
        self.min_norm = 0.1
        self.max_norm = self.max_xy * np.sqrt(3)

    def data_to_formatted_rgb(self, data):
        data_rgb = [[color_fader_rgb255(self.c1, self.c2,
                                        mix=min(1, np.linalg.norm(data[column][row])/self.max_norm)) for row in range(column_len)]
                    for column, column_len in enumerate(self.grid_size)]
        return data_rgb

    def update_view(self, data):
        for row in range(self.view.grid_size[0]):
            for column in range(self.view.grid_size[1]):
                data[row][column][0] = max(min(data[row][column][0], self.max_xy), -self.max_xy)
                data[row][column][1] = max(min(data[row][column][1], self.max_xy), -self.max_xy)
                data[row][column][2] = min(data[row][column][2], np.sqrt(self.max_norm**2 - data[row][column][0]**2 - data[row][column][1]**2))
                self.view.circle_radii[row][column] = np.linalg.norm(data[row][column])/self.max_norm * self.view.radius_max + self.view.radius_min
                self.view.circle_offsets[row][column] = (data[row][column][0]/self.max_xy * self.view.offset_max,
                                                         data[row][column][1]/self.max_xy * self.view.offset_max)
        self.view.circle_colors = self.data_to_formatted_rgb(data)
        self.view.update_view()
