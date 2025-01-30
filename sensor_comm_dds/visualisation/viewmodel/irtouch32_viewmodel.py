import asyncio
import time
import numpy as np
import scipy.optimize as scipy_opt
from loguru import logger

from sensor_comm_dds.visualisation.view.irtouch32_view import IRTouch32View


class IRTouch32ViewModel:
    def __init__(self, view):
        self.view: IRTouch32View = view
        self.grid_size = self.view.grid_size
        self.num_taxels = 0
        for column_len in self.grid_size:
            self.num_taxels += column_len
        self.calibration_data = [255 for _ in range(self.num_taxels)]
        self.current_data = [0 for _ in range(self.num_taxels)]
        self.calibrated_data = [0 for _ in range(self.num_taxels)]
        self.formatted_data = [[0 for _ in range(column_len)] for column_len in self.grid_size]
        self.grid_values_mean = None

        self.darkness_threshold = -1
        self.num_active_edge_markers = 0
        self.num_dark = 32

        self.edge_fit_error = 0
        self.corner_fit_angle = 0
        self.init_corner_fit = [self.view.cell_width + self.view.tile_width / 2,
                                self.view.cell_height + self.view.tile_height / 2,
                                180, 270]
        self.prev_corner_fit = self.init_corner_fit.copy()

    def update_data(self, data):
        self.current_data = data
        self.calibrate_data()
        self.format_data()
        self.view.hex_values = self.formatted_data
        self.update_edge_detection()

    def update(self, data):
        self.update_data(data)
        self.view.update_view()

    def update_edge_detection(self):
        self.update_darkness_threshold()
        #self.update_darkness_centrepoint()
        self.update_edge_markers()
        # self.update_corner_fit()
        #self.update_edge_fit()
        # t_now = time.time()
        # logger.debug(f'Update freq: {1/(t_now - self.last_time_since_edge_detection)}')
        # self.last_time_since_edge_detection = t_now

    def update_darkness_threshold(self):
        data_sorted = np.array(sorted(self.calibrated_data))
        self.grid_values_mean = np.mean(data_sorted)
        data_sorted_low = data_sorted[:-1]
        data_sorted_high = data_sorted[1:]
        data_diff = abs(data_sorted_high - data_sorted_low)
        cutoff_idx = np.argmax(data_diff)
        self.num_dark = cutoff_idx + 1
        cluster_dark_mean = np.mean(data_sorted[:cutoff_idx + 1])
        cluster_bright_mean = np.mean(data_sorted[cutoff_idx + 1:])
        if abs(cluster_dark_mean - cluster_bright_mean) < 80 or cluster_bright_mean < 100:
            self.darkness_threshold = -1
        else:
            self.darkness_threshold = (cluster_bright_mean + cluster_dark_mean) / 2

    def update_darkness_centrepoint(self):
        if self.darkness_threshold == -1:
            self.view.darkness_centrepoint = None
            return
        x_centrepoint = 0
        y_centrepoint = 0
        num_dark_hexagons = 0
        normalisation = 0
        for column, column_len in enumerate(self.grid_size):
            for row in range(column_len):
                value = self.formatted_data[column][row]
                if value < self.darkness_threshold:
                    num_dark_hexagons += 1
                    weight = (255 - value)
                    normalisation += weight
                    hexagon_centrepoint = self.view.hexagon_centrepoints[row, column]
                    x_centrepoint += weight * hexagon_centrepoint[0]
                    y_centrepoint += weight * hexagon_centrepoint[1]
        if not normalisation:
            return
        x_centrepoint /= normalisation
        y_centrepoint /= normalisation
        self.view.darkness_centrepoint = [x_centrepoint, y_centrepoint]

    def update_edge_markers(self):
        """
        TODO: prettify (as-is the code does avoid if-statements in loops, improving performance, but bulking up the code)
        """
        self.view.edge_marker_centrepoints = []
        self.num_active_edge_markers = 0
        if self.darkness_threshold == -1:
            return
        for column, column_len in enumerate(self.grid_size[:-1]):
            next_column_len = self.grid_size[column + 1]
            if column_len > next_column_len:
                value_centre = self.formatted_data[column][0]
                value_right_down = self.formatted_data[column + 1][0]
                value_down = self.formatted_data[column][1]

                centrepoint_centre = self.view.hexagon_centrepoints[0, column]
                centrepoint_right_down = self.view.hexagon_centrepoints[0, column + 1]
                centrepoint_down = self.view.hexagon_centrepoints[1, column]

                self.evaluate_edge_and_create_marker(centrepoint_centre, centrepoint_right_down, value_centre,
                                                     value_right_down)
                self.evaluate_edge_and_create_marker(centrepoint_centre, centrepoint_down, value_centre,
                                                     value_down)
                for row in range(1, column_len - 1):
                    value_centre = self.formatted_data[column][row]
                    value_right_down = self.formatted_data[column + 1][row]
                    value_right_up = self.formatted_data[column + 1][row - 1]
                    value_down = self.formatted_data[column][row + 1]

                    centrepoint_centre = self.view.hexagon_centrepoints[row, column]
                    centrepoint_right_down = self.view.hexagon_centrepoints[row, column + 1]
                    centrepoint_right_up = self.view.hexagon_centrepoints[row - 1, column + 1]
                    centrepoint_down = self.view.hexagon_centrepoints[row + 1, column]

                    self.evaluate_edge_and_create_marker(centrepoint_centre, centrepoint_right_up, value_centre,
                                                         value_right_up)
                    self.evaluate_edge_and_create_marker(centrepoint_centre, centrepoint_right_down,
                                                         value_centre, value_right_down)
                    self.evaluate_edge_and_create_marker(centrepoint_centre, centrepoint_down, value_centre,
                                                         value_down)

                value_centre = self.formatted_data[column][-1]
                value_right_up = self.formatted_data[column + 1][-1]

                centrepoint_centre = self.view.hexagon_centrepoints[column_len - 1, column]
                centrepoint_right_up = self.view.hexagon_centrepoints[column_len - 2, column + 1]

                self.evaluate_edge_and_create_marker(centrepoint_centre, centrepoint_right_up, value_centre,
                                                     value_right_up)
            if column_len < next_column_len:
                for row in range(column_len - 1):
                    value_centre = self.formatted_data[column][row]
                    value_right_up = self.formatted_data[column + 1][row]
                    value_right_down = self.formatted_data[column + 1][row + 1]
                    value_down = self.formatted_data[column][row + 1]

                    centrepoint_centre = self.view.hexagon_centrepoints[row, column]
                    centrepoint_right_up = self.view.hexagon_centrepoints[row, column + 1]
                    centrepoint_right_down = self.view.hexagon_centrepoints[row + 1, column + 1]
                    centrepoint_down = self.view.hexagon_centrepoints[row + 1, column]

                    self.evaluate_edge_and_create_marker(centrepoint_centre, centrepoint_right_up, value_centre,
                                                         value_right_up)
                    self.evaluate_edge_and_create_marker(centrepoint_centre, centrepoint_right_down,
                                                         value_centre,
                                                         value_right_down)
                    self.evaluate_edge_and_create_marker(centrepoint_centre, centrepoint_down, value_centre,
                                                         value_down)
                value_centre = self.formatted_data[column][column_len - 1]
                value_right_up = self.formatted_data[column + 1][column_len - 1]
                value_right_down = self.formatted_data[column + 1][column_len]

                centrepoint_centre = self.view.hexagon_centrepoints[column_len - 1, column]
                centrepoint_right_up = self.view.hexagon_centrepoints[column_len - 1, column + 1]
                centrepoint_right_down = self.view.hexagon_centrepoints[column_len, column + 1]

                self.evaluate_edge_and_create_marker(centrepoint_centre, centrepoint_right_up, value_centre,
                                                     value_right_up)
                self.evaluate_edge_and_create_marker(centrepoint_centre, centrepoint_right_down, value_centre,
                                                     value_right_down)
            else:
                Exception("Unexpected hexagonal grid definition: adjacent columns of equal size")
        return

    def evaluate_edge_and_create_marker(self, centrepoint1, centrepoint2, value1, value2):
        centrepoint1 = np.array(centrepoint1)
        centrepoint2 = np.array(centrepoint2)
        if min(value1, value2) <= self.darkness_threshold < max(value1, value2):
            # indicates an edge between the two centrepoints
            value_difference = value2 - value1
            direction_vector = centrepoint2 - centrepoint1
            distance = np.linalg.norm(direction_vector)
            direction_vector /= distance
            edge_marker_coords = centrepoint1 + direction_vector * distance * (
                    self.darkness_threshold - value1) / value_difference
            self.view.edge_marker_centrepoints.append(edge_marker_coords)
            self.num_active_edge_markers += 1

    def edge_fitting_function(self, xy, y0, angle):
        # TODO: add parameter for x0?
        x0 = self.view.edge_x0
        p0 = np.array([x0, y0])
        xy = np.array(xy)
        v = [np.cos(angle * np.pi / 180), np.sin(angle * np.pi / 180)]  # vector defined by min_angle
        distance_xy_v = abs(np.cross(v, xy - p0) / np.linalg.norm(v))
        return distance_xy_v

    def update_edge_fit(self):
        xy_array = np.array(self.view.edge_marker_centrepoints)
        x0 = self.view.cell_width + self.view.tile_width / 2
        self.view.edge_params = [x0, 0, 0]
        if xy_array.shape[0] < 2:
            self.edge_fit_error = 0  # no edge so no error
        else:
            z_array = np.array([0 for _ in range(len(self.view.edge_marker_centrepoints))])
            guess = [self.view.cell_height + self.view.tile_height / 2, 0]
            try:
                optimisation_result = scipy_opt.curve_fit(self.edge_fitting_function, xdata=xy_array,
                                                          ydata=z_array, p0=guess)
                self.view.edge_params[1:] = optimisation_result[0]
                perr = np.sqrt(np.diag(optimisation_result[1]))
                self.edge_fit_error = np.mean(perr)
                # logger.debug(f'Edge fit error: {self.edge_fit_errors[device]}')
            except RuntimeError:
                self.view.edge_params = [x0, 0, 0]
                logger.error("Edge fit did not find a solution")

    @staticmethod
    def corner_fitting_function(xy, x0=30, y0=20, angle1=120, angle2=335):
        """  y
             ^    _________________     _________________           _________________
             |   |      /         |    |      |         |          |    \\          |
             |   |_____/(x0, y0)  |    |     /(x0, y0)  |          |_____\\(x0, y0) |
             |   |________________|    |___ /___________|          |________________|
             |__________________________________________________________________________> x

        The edge of the cloth is approximated by 2 straight lines as indicated above. angle1 and angle2 are angles (in degrees!)
        with respect to positive x, counterclockwise (CCW) being a positive angle, fixing the orientation of the two lines
        around the point (x0, y0)
        """
        # TODO: evaluate RMLS (robust moving least squares)
        p0 = np.array([x0, y0])
        angle1 = angle1 % 360
        angle2 = angle2 % 360
        min_angle = min(angle1, angle2)
        max_angle = max(angle1, angle2)
        xy = np.array(xy)
        v_p0_xy = xy - p0  # vectors defined from p0 to xy points
        v_min = np.array(
            [np.cos(min_angle * np.pi / 180), np.sin(min_angle * np.pi / 180)])  # vector defined by min_angle
        v_max = np.array(
            [np.cos(max_angle * np.pi / 180), np.sin(max_angle * np.pi / 180)])  # vector defined by max_angle
        t_min = v_min @ np.transpose(v_p0_xy)
        t_min = np.maximum(0, t_min)
        t_max = v_max @ np.transpose(v_p0_xy)
        t_max = np.maximum(0, t_max)
        d_min = np.linalg.norm(p0 + t_min.reshape(-1, 1) * v_min - xy,
                               axis=1)  # distances from xy points to half-infinite line defined by p0 and min_angle
        d_max = np.linalg.norm(p0 + t_max.reshape(-1, 1) * v_max - xy,
                               axis=1)  # distances from xy points to half-infinite line defined by p0 and max_angle
        return np.minimum(d_min, d_max)

    @staticmethod
    def angle_wrt_pos_x(p0, p1):
        """
        p0 and p1[i] define a vector from p0 to p1[i], the function returns the angle of this vector w.r.t. positive x in the
        range [0, 360[
        """
        angle = np.arctan2((p1[:, 1] - p0[1]), (p1[:, 0] - p0[0])) * 180 / np.pi  # angle of the xy point wrt (x0, y0)
        angle += (angle < 0) * 360
        return angle

    def update_corner_fit(self):
        self.view.corner_params = [0, 0, 0, 0]
        xy_array = np.array(self.view.edge_marker_centrepoints)
        if xy_array.shape[0] < 2:
            pass
        else:
            z_array = np.array([0 for _ in range(len(self.view.edge_marker_centrepoints))])
            bounds = ([self.view.cell_width,
                       self.view.cell_height, 0, 0],
                      [self.view.cell_width + self.view.tile_width,
                       self.view.cell_height + self.view.tile_height, 450, 450]
                      )
            educated_guess = np.array(self.prev_corner_fit)
            init_guess_x0 = self.init_corner_fit[0]
            init_guess_y0 = self.init_corner_fit[1]
            guesses = [
                educated_guess,
                [init_guess_x0, init_guess_y0, 180, 270],
                [init_guess_x0, init_guess_y0, 0, 90],
                [init_guess_x0, init_guess_y0, 90, 180],
                [init_guess_x0, init_guess_y0, 270, 360],
            ]
            self.view.corner_params = [0, 0, 0, 0]
            error_prev = np.inf
            for guess in guesses:
                try:
                    optimisation_result = scipy_opt.curve_fit(IRTouch32ViewModel.corner_fitting_function,
                                                              xdata=xy_array,
                                                              ydata=z_array,
                                                              p0=guess,
                                                              bounds=bounds)
                    perr = np.sqrt(np.diag(optimisation_result[1]))
                    error = np.mean(perr)
                    if error < 15:
                        parameters = optimisation_result[0]
                        self.view.corner_params = parameters
                        angle = abs(parameters[3] - parameters[2])
                        while angle > 180:
                            angle -= 360
                        self.corner_fit_angle = abs(angle)
                        error_prev = error
                        self.prev_corner_fit = list(self.view.corner_params)
                        break
                except RuntimeError:
                    self.view.corner_params = [0, 0, 0, 0]
                    logger.error("Corner fit did not find a solution")
            else:
                logger.error("No corner fit with low enough error was found")
            logger.debug(f'Corner fit error: {error_prev}')

    def calibrate_data(self):
        # self.calibrated_data[device] = 255 - np.rint(abs(np.array(self.calibration_data) - np.array(data))).astype(int)
        self.calibrated_data = np.clip(np.rint(np.array(self.current_data)
                                               / np.array(self.calibration_data) * 255).astype(int),
                                       0, 255)

    def format_data(self):
        """
        Data is a list of 32 values, with the list index of a value corresponding to the grid index as defined in the
        __init__ of HexGridPlot.
        """
        ctr = 0
        for column, column_len in enumerate(self.grid_size):
            for row in range(column_len):
                self.formatted_data[column][row] = self.calibrated_data[ctr]
                ctr += 1
