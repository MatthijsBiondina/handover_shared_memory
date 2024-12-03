import numpy as np


def make_pixel_grid(shape):
    h, w = shape[:2]

    X, Y = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
    return np.stack((X, Y), axis=-1)
