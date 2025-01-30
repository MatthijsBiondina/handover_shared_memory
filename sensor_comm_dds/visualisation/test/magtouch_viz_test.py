from sensor_comm_dds.visualisation.view.bubble_view import BubbleView
from sensor_comm_dds.visualisation.viewmodel.magtouch_viewmodel import MagTouchViewModel
import time
import numpy as np
import math


device = 'test'
view = BubbleView(name=device, grid_size=(2, 2))
viewmodel = MagTouchViewModel(view=view, c1='#434A52', c2='#7DB5A8')
view.circle_radii = [[20 for j in range(view.grid_size[1])]
                                          for k in range(view.grid_size[0])]
view.circle_offsets = [[[0, 0] for j in range(view.grid_size[1])]
                               for k in range(view.grid_size[0])]
view.circle_offsets[0][0] = [-view.offset_max, view.offset_max]
view.circle_offsets[0][1] = [view.offset_max, view.offset_max]
data = np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
data = np.array([[[viewmodel.max_xy-1 for _ in range(3)] for _ in range(2)] for _ in range(2)])
data[0, 0] = [-viewmodel.max_xy+1 for _ in range(3)]

while True:
    data = np.fmod(data + 1, viewmodel.max_xy)
    viewmodel.update_view(data)
    time.sleep(0.01)