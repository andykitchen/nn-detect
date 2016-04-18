import numpy as np
import skimage

def random_polygon(w=10, h=10, sides=5):
    img = np.zeros((h, w), dtype=np.uint8)
    x = np.random.randint(0, w, size=sides)
    y = np.random.randint(0, h, size=sides)
    rr, cc = skimage.draw.polygon(y, x)
    img[rr, cc] = 1
    return img

grass_image = skimage.io.imread('grass.jpg')
stones_image = skimage.io.imread('stones.jpg')

def random_textured_polygon(w=256, h=256, sides=5):
	stones_small = skimage.transform.resize(stones_image[:,:stones_image.shape[0]], (h, w))
	grass_small = skimage.transform.resize(grass_image[:,:grass_image.shape[0]], (h, w))

	mask = random_polygon(w, h, 5)

	mask3 = mask[:,:,np.newaxis]
	img = mask3*grass_small + (1 - mask3)*stones_small
	return img, mask
