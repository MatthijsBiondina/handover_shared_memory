import numpy as np
from sensor_comm_dds.visualisation.viz_utils import color_fader_rgb255


class ClotheshangerViewModel:
    def __init__(self, view, c1='#434A52', c2='#D6AE72'):
        self.view = view
        self.grid_size = self.view.grid_size
        self.num_taxels = self.grid_size[1]
        self.c1 = c1
        self.c2 = c2

    def data_to_formatted_rgb(self, data):
        data_rgb = [[color_fader_rgb255(self.c1, self.c2, mix=data[idx]/256) for idx in range(self.num_taxels)]]
        return data_rgb

    def update_view(self, data):
        self.view.square_colors = self.data_to_formatted_rgb(data)
        self.view.text = {(0, column): str(data[column]) for column in range(self.num_taxels)}
        self.view.update_view()
