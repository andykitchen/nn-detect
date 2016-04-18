import numpy as np
from skimage.draw import polygon

def random_polygon(w=10, h=10, sides=5):
    img = np.zeros((w, h), dtype=np.uint8)
    x = np.random.randint(0, w, size=sides)
    y = np.random.randint(0, h, size=sides)
    rr, cc = polygon(y, x)
    img[rr, cc] = 1
    return img
