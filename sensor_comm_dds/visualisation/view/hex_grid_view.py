import tkinter as tk
import numpy as np
import matplotlib as mpl
from loguru import logger
from sensor_comm_dds.visualisation.view.view import View
from sensor_comm_dds.visualisation.viz_utils import color_fader_rgb255


class HexGridView(View):
    def __init__(self, name=None, grid_size=(5, 4, 5, 4, 5, 4, 5), disp_vals=True,
                 c1='#000000', c2='#FFFFFF'):
        """
        param grid_size: indicates the amount of cells for each column
        Cell indexing:
           _ _           _ _
         /     \\      /     \\
        /   0   \\_ _ /   5   \\
        \\      /     \\      /
         \\_ _ /   3   \\_ _ /
         /     \\      /     \\
        /   1   \\_ _ /   6   \\
        \\      /     \\      /
         \\_ _ /   4   \\_ _ /
         /     \\      /     \\
        /   2   \\_ _ /   7   \\
        \\      /     \\      /
         \\_ _ /       \\_ _ /
        """
        self.grid_size = grid_size
        self.cell_circumradius = 50  # the radius of de circle circumscribing the hexagonal cells
        self.cell_inradius = self.cell_circumradius * np.sqrt(3) / 2  # the radius of de circle inscribing the
        # hexagonal cells
        self.cell_width = 2 * self.cell_circumradius
        self.cell_height = 2 * self.cell_inradius
        self.tile_width = (len(self.grid_size) * 3 / 4 + 1 / 4) * self.cell_width
        self.tile_height = max(self.grid_size) * self.cell_height
        super().__init__(canvas_width=(self.tile_width + self.cell_width) + self.cell_width,
                         canvas_height=self.tile_height * 1.5)

        self.disp_vals = disp_vals
        self.c1 = c1
        self.c2 = c2
        self.canvas.create_text(
            self.cell_width + self.tile_width * 1 / 2,
            (max(self.grid_size) + 1 + 1 / 2) * self.cell_height, fill="#000000", font="Arial 20 bold",
            text=name)

        self.hex_values = [[0 for _ in range(column_len)]
                           for column_len in self.grid_size]
        self.hex_colors = [[[0 for _ in range(3)] for _ in range(column_len)]
                           for column_len in self.grid_size]

        # Create individual hexagons
        self.hexagon_ids = {}
        self.hexagon_centrepoints = {}
        self.text_ids = {}
        self._create_hexagons()

    def _create_hexagons(self):
        for column, column_len in enumerate(self.grid_size):
            for row in range(column_len):
                x_centre = 1 / 2 * self.cell_width + self.cell_width + 3 / 4 * column * self.cell_width
                y_centre = self.cell_height + (row + 1) * self.cell_height - (column_len % 2) * self.cell_height / 2
                self.hexagon_centrepoints[row, column] = (x_centre, y_centre)
                self.hexagon_ids[row, column] = self._create_hexagon_from_centrepoint(x_centre, y_centre)
                if self.disp_vals:
                    self.text_ids[row, column] = self.canvas.create_text(x_centre, y_centre, fill="#ededed",
                                                                         font="Arial 20 bold", text='rgb')

    def _create_hexagon_from_centrepoint(self, x_center, y_center):
        angle = np.pi / 3
        coords = []
        for i in range(6):
            x = x_center + self.cell_circumradius * np.cos(angle * i)
            y = y_center + self.cell_circumradius * np.sin(angle * i)
            coords.append([x, y])
        hexagon = self.canvas.create_polygon(coords[0][0],
                                             coords[0][1],
                                             coords[1][0],
                                             coords[1][1],
                                             coords[2][0],
                                             coords[2][1],
                                             coords[3][0],
                                             coords[3][1],
                                             coords[4][0],
                                             coords[4][1],
                                             coords[5][0],
                                             coords[5][1], fill="#ededed", tags="hex")
        return hexagon

    def redraw(self):
        self.update_grid_colors_from_values()
        for column, column_len in enumerate(self.grid_size):
            for row in range(column_len):
                item_id_hex = self.hexagon_ids[row, column]
                value = self.hex_values[column][row]
                rgb = self.hex_colors[column][row]
                self.canvas.itemconfig(item_id_hex, fill="#%02x%02x%02x" % tuple(rgb))
                if self.disp_vals:
                    item_id_text = self.text_ids[row, column]
                    hsv = mpl.colors.rgb_to_hsv(rgb)
                    if hsv[2] < 150:
                        self.canvas.itemconfig(item_id_text, text=str(value), fill="#ededed")
                    else:
                        self.canvas.itemconfig(item_id_text, text=str(value), fill="#212121")

    def update_grid_colors_from_values(self):
        data_rgb = [[color_fader_rgb255(self.c1, self.c2, mix=self.hex_values[column][row] / 255) for row in
                     range(column_len)]
                    for column, column_len in enumerate(self.grid_size)]
        self.hex_colors = data_rgb
