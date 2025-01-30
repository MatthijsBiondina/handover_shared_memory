import numpy as np


class SwitchesViewModel:
    def __init__(self, view, c1='#434A52', c2='#6595BF'):
        self.view = view
        self.grid_size = self.view.grid_size
        self.c1 = c1
        self.c2 = c2

    def data_to_formatted_rgb(self, data):
        """
        Converts data from a single sensor to a proper RGB format for the GridPlot
        """
        data_rgb = []
        for val in data:
            if val == 1:
                data_rgb.append([0, 0, 0])
            else:
                data_rgb.append([255, 255, 255])
        return [data_rgb]

    def update_view(self, data):
        self.view.square_colors = self.data_to_formatted_rgb(data)
        #self.view.text = {(column, row): str(data[1 + row * self.grid_size[1] + column]) for row in range(self.grid_size[0])
        #                  for column in range(self.grid_size[1])}
        self.view.update_view()
