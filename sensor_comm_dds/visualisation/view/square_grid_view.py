import tkinter as tk
from loguru import logger

from sensor_comm_dds.visualisation.view.view import View
from sensor_comm_dds.visualisation.viz_utils import constrain_rgb


class SquareGridView(View):
    """
    Based on: https://stackoverflow.com/questions/4781184/tkinter-displaying-a-square-grid
    This class will plot RGB (8 bit!) data in a grid.
    """

    def __init__(self, name=None, grid_size=(7, 7), disp_vals=False, cell_width=70):
        self.cell_width = cell_width
        self.cell_height = cell_width
        self.grid_size = grid_size
        self.disp_vals = disp_vals
        tile_width = self.grid_size[1] * self.cell_width
        tile_height = self.grid_size[0] * self.cell_height + 1.5 * self.cell_height
        super().__init__(canvas_width=(tile_width + self.cell_width) + self.cell_width,
                         canvas_height=tile_height * 1.5)

        # Write name under respective tile.
        self.canvas.create_text(self.cell_width + tile_width * 1 / 2,
                                (self.grid_size[0] + 1 + 1 / 2) * self.cell_height, fill="#000000",
                                font="Arial 20 bold",
                                text=name)

        self.square_colors = [[[0 for _ in range(3)] for __ in range(self.grid_size[1])]
                              for ___ in range(self.grid_size[0])]

        # Create individual squares
        self.rect_ids = {}
        self.text_ids = {}
        self.text = {(row, column): str(0) for row in range(self.grid_size[0]) for column in range(self.grid_size[1])}
        self._create_individual_squares()

        self.redraw()

    def _create_individual_squares(self):
        for row in range(self.grid_size[0]):
            for column in range(self.grid_size[1]):
                x1 = self.cell_width + column * self.cell_width
                y1 = self.cell_height + row * self.cell_height
                x2 = x1 + self.cell_width
                y2 = y1 + self.cell_height
                self.rect_ids[row, column] = self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", tags="rect")
                if self.disp_vals:
                    self.text_ids[row, column] = self.canvas.create_text(x1 + self.cell_width / 2, y1 +
                                                                         self.cell_height / 2, fill="#ededed",
                                                                         font="Arial 20 bold", text='rgb')

    def redraw(self):
        # try:
        for row in range(self.grid_size[0]):
            for column in range(self.grid_size[1]):
                item_id_rect = self.rect_ids[row, column]
                rgb = constrain_rgb(self.square_colors[row][column])
                self.canvas.itemconfig(item_id_rect, fill="#%02x%02x%02x" % tuple(rgb))
                if self.disp_vals:
                    item_id_text = self.text_ids[row, column]
                    if rgb[0] < 128:
                        self.canvas.itemconfig(item_id_text, text=self.text[row, column], fill="#ededed")
                    else:
                        self.canvas.itemconfig(item_id_text, text=self.text[row, column], fill="#212121")

        # except Exception as e:  # TODO: handle properly
        #    logger.error(f'Redraw failed: {e}')
