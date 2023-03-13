import numpy as np


def uniform_attachment_graphon(x, y):
    return 1 - np.maximum(x, y)


def ranked_attachment_graphon(x, y):
    return 1 - x * y


def er_graphon(x, y, p=0.5):
    return np.ones_like(x) * p


def power_law_graphon(x, y, alpha=0.5):
    return (1-alpha)**2 * np.power(x * y, -alpha)


def cutoff_power_law_graphon(x, y, alpha=0.5, c=0.1):
    return ((1-alpha) / (1-alpha * np.power(c, 1-alpha)))**2 * np.power(np.maximum(x,c) * np.maximum(y,c), -alpha)
