from loguru import logger
from sensor_comm_dds.visualisation.view.view import View


class BubbleView(View):
    def __init__(self, name=None, grid_size=(2, 2), disp_vals=False):
        self.radius_min = 20
        self.radius_max = 60
        self.offset_max = self.radius_max
        self.grid_size = grid_size
        self.tile_width = self.grid_size[1] * 2 * (self.offset_max + self.radius_max)
        self.tile_height = self.grid_size[0] * 2 * (self.offset_max + self.radius_max)
        super().__init__(canvas_width=self.tile_width + 2 * self.radius_max,
                         canvas_height=self.tile_height + 2 * self.radius_max)
        
        self.disp_vals = disp_vals
        self.canvas.create_text(self.radius_max + self.tile_width / 2,
                                self.tile_height + self.radius_max * 3 / 2, fill="#000000", font="Arial 20 bold",
                                text=name)
        self.circle_radii = [[0 for j in range(self.grid_size[1])]
                             for k in range(self.grid_size[0])]
        self.circle_offsets = [[(0, 0) for j in range(self.grid_size[1])]
                               for k in range(self.grid_size[0])]
        self.circle_colors = [[(0, 0, 0) for j in range(self.grid_size[1])]
                              for k in range(self.grid_size[0])]
        # Create individual circles & lines
        self.circles = [[None for j in range(self.grid_size[1])]
                        for k in range(self.grid_size[0])]
        self.text = [[None for j in range(self.grid_size[1])]
                     for k in range(self.grid_size[0])]
        self._create_individual_circles()
        self.lines = [[None for j in range(self.grid_size[1])]
                      for k in range(self.grid_size[0])]
        self._create_individual_lines()

        self.redraw()

    def _create_circle(self, x, y, r, **kwargs):
        return self.canvas.create_oval(x - r, y - r, x + r, y + r, **kwargs)

    def _create_individual_circles(self):
        for row in range(self.grid_size[0]):
            for column in range(self.grid_size[1]):
                x = self.radius_max + self.offset_max
                y = self.radius_max + self.offset_max
                self.circles[row][column] = self._create_circle(x, y, self.radius_min, fill="black",
                                                                tags="oval")
                if self.disp_vals:
                    self.text[row][column] = self.canvas.create_text(x, y, fill="#ededed",
                                                                     font="Arial 20 bold", text='rgb')

    def _create_individual_lines(self, width=3):
        for row in range(self.grid_size[0]):
            for column in range(self.grid_size[1]):
                self.lines[row][column] = self.canvas.create_line(0, 0, 0, 0, fill="#D6AE72", width=width)

    def redraw(self):
        for row in range(self.grid_size[0]):
            for column in range(self.grid_size[1]):
                circle_id = self.circles[row][column]
                line_id = self.lines[row][column]
                base_x = self.radius_max + (self.offset_max + self.radius_max) + column * 2 * (self.offset_max + self.radius_max)
                x = base_x + self.circle_offsets[row][column][0]
                base_y = self.radius_max + (self.offset_max + self.radius_max) + row * 2 * (self.offset_max + self.radius_max)
                y = base_y + self.circle_offsets[row][column][1]
                r = max(min(self.circle_radii[row][column], self.radius_max), self.radius_min)
                new_line_coords = (base_x, base_y, x, y)
                new_circle_coords = (x - r, y - r, x + r, y + r)
                self.canvas.coords(line_id, *new_line_coords)
                self.canvas.coords(circle_id, *new_circle_coords)
                self.canvas.itemconfig(circle_id, fill="#%02x%02x%02x" % tuple(self.circle_colors[row][column]))
                if self.disp_vals:
                    item_id_text = self.text[row][column]
