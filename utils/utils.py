import numpy as np


def normalize_angle(angle):
    while angle > np.pi:
        angle -= np.pi

    while angle <= -np.pi:
        angle += np.pi

    return angle
