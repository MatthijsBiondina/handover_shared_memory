import matplotlib as mpl
import numpy as np
from loguru import logger


def color_fader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    if mix > 1:
        logger.warning("color_fader received mix value > 1")
        mix = 1
    if mix < 0:
        logger.warning("color_fader received mix value < 0")
        mix = 0
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_rgb((1 - mix) * c1 + mix * c2)


def color_fader_rgb255(c1, c2, mix=0):
    rgb1 = color_fader(c1, c2, mix=mix)
    rgb255 = [int(rgb1[i] * 255) for i in range(len(rgb1))]
    return rgb255


def constrain_rgb(rgb):
    for i in range(3):
        if rgb[i] > 255:
            logger.warning("GridPlot received RGB value > 255")
            rgb[i] = 255
        elif rgb[i] < 0:
            logger.warning("GridPlot received RGB value < 0")
            rgb[i] = 0
    return rgb
