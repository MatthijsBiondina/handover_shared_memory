import tkinter
import numpy as np
from loguru import logger
from sensor_comm_dds.visualisation.view.hex_grid_view import HexGridView


class IRTouch32View(HexGridView):
    def __init__(self, name=None, grid_size=(5, 4, 5, 4, 5, 4, 5), disp_vals=True, c1='#434A52', c2='#7DB5A8'):
        super().__init__(name=name, grid_size=grid_size, disp_vals=disp_vals, c1=c1, c2=c2)

        def edge_marker():
            edge_marker = self.canvas.create_oval(
                0 - self.edge_marker_width / 2,
                0 - self.edge_marker_width / 2,
                0 + self.edge_marker_width / 2,
                0 + self.edge_marker_width / 2,
                fill="#D6AE72", state=tkinter.HIDDEN
            )
            return edge_marker

        self.edge_marker_width = 10
        self.num_edge_markers = 3 * 12 + 3 * 11  # TODO: derive from grid_size
        self.edge_markers = [edge_marker() for _ in range(self.num_edge_markers)]
        # edge marker items are preallocated and made visible/positioned as needed
        self.edge_marker_centrepoints = []

        def edge(width=1):
            edge = self.canvas.create_line(0, 0, 0, 0, fill="#D6AE72", width=width)
            return edge

        self.edge = edge(width=3)
        self.edge_len = len(self.grid_size) * self.cell_width
        self.edge_params = [0, 0, 0]

        self.corner_edges = [edge() for _ in range(2)]
        self.corner_params = [0, 0, 0, 0]

        self.edge_x0 = self.cell_width + self.tile_width / 2
        self.darkness_centrepoint = (0, 0)
        self.darkness_centrepoint_width = 20
        self.darkness_centrepoint_id = self.canvas.create_oval(-self.darkness_centrepoint_width / 2,
                                                               -self.darkness_centrepoint_width / 2,
                                                               self.darkness_centrepoint_width,
                                                               self.darkness_centrepoint_width,
                                                               fill="#6595BF")
        self.redraw()

    def redraw(self):
        for marker in self.edge_markers:
            self.canvas.itemconfig(marker, state=tkinter.HIDDEN)

        if self.darkness_centrepoint:
            self.canvas.moveto(self.darkness_centrepoint_id,
                               x=self.darkness_centrepoint[0] - self.darkness_centrepoint_width / 2,
                               y=self.darkness_centrepoint[1] - self.darkness_centrepoint_width / 2)
            self.canvas.itemconfig(self.darkness_centrepoint_id, state=tkinter.NORMAL)
            # Edge markers
            for i, edge_marker_coords in enumerate(self.edge_marker_centrepoints):
                edge_marker = self.edge_markers[i]
                self.canvas.itemconfig(edge_marker, state=tkinter.NORMAL)
                self.canvas.moveto(edge_marker, edge_marker_coords[0] - self.edge_marker_width / 2,
                                   edge_marker_coords[1] - self.edge_marker_width / 2)

            # Straight edges
            x0, y0, angle = self.edge_params
            self.canvas.coords(self.edge, x0 - self.edge_len * np.cos(angle * np.pi / 180) / 2,
                               y0 - self.edge_len * np.sin(angle * np.pi / 180) / 2,
                               x0 + self.edge_len * np.cos(angle * np.pi / 180) / 2,
                               y0 + self.edge_len * np.sin(angle * np.pi / 180) / 2)
            self.canvas.itemconfig(self.edge, state=tkinter.NORMAL)

            # Corner edges
            edge_len = 300
            x0, y0, angle1, angle2 = self.corner_params
            self.canvas.coords(self.corner_edges[0], x0, y0, x0 + edge_len * np.cos(angle1 * np.pi / 180),
                               y0 + edge_len * np.sin(angle1 * np.pi / 180))
            self.canvas.coords(self.corner_edges[1], x0, y0, x0 + edge_len * np.cos(angle2 * np.pi / 180),
                               y0 + edge_len * np.sin(angle2 * np.pi / 180))
            self.canvas.itemconfig(self.corner_edges[0], state=tkinter.NORMAL)
            self.canvas.itemconfig(self.corner_edges[1], state=tkinter.NORMAL)
        else:
            self.canvas.itemconfig(self.darkness_centrepoint_id, state=tkinter.HIDDEN)
            self.canvas.itemconfig(self.edge, state=tkinter.HIDDEN)
            self.canvas.itemconfig(self.corner_edges[0], state=tkinter.HIDDEN)
            self.canvas.itemconfig(self.corner_edges[1], state=tkinter.HIDDEN)

        super(IRTouch32View, self).redraw()
