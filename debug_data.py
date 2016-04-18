import numpy as np
import skimage
import skimage.io
import skimage.draw
import skimage.transform
import skimage.morphology
import skimage.color

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
	chull = skimage.morphology.convex_hull_image(mask)
	mask[chull] = 1

	mask = mask[:,:,np.newaxis]
	img = mask*grass_small + (1 - mask)*stones_small
	return img, mask

def generate_batch(w=128, h=128, minibatch_size=10, sides=5):
	input_data = np.zeros([minibatch_size, h, w, 1])
	label_data = np.zeros([minibatch_size, h, w, 1], dtype=np.int64)

	for i in range(minibatch_size):
		img, mask = random_textured_polygon(w, h, sides)
		img_gray = skimage.color.rgb2gray(img)
		input_data[i, :, :, 0] = 2*(img_gray / 255) - 1
		label_data[i, :, :] = mask

	return input_data, label_data
