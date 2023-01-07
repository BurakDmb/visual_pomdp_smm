import numpy as np
from PIL import Image


def resize_obs(obs, pixel_size):
    obs_resized = np.array(Image.fromarray(obs).resize(
        (pixel_size, pixel_size)))
    return obs_resized
