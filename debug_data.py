import numpy as np
import skimage
import skimage.io
import skimage.draw
import skimage.transform
import skimage.morphology
import skimage.color

def random_polygon(w=128, h=128, sides=5):
    img = np.zeros((h, w), dtype=np.uint8)
    x = np.random.randint(0, w, size=sides)
    y = np.random.randint(0, h, size=sides)
    rr, cc = skimage.draw.polygon(y, x)
    img[rr, cc] = 1
    return img


def random_textured_polygon(fg, bg, w=128, h=128, sides=5):
	mask = random_polygon(w, h, 5)
	chull = skimage.morphology.convex_hull_image(mask)
	mask[chull] = 1

	mask = mask[:,:,np.newaxis]
	img = mask*fg + (1 - mask)*bg
	return img, mask


def load_default_textures(w, h):
	grass_image  = skimage.io.imread('resources/grass.jpg')
	stones_image = skimage.io.imread('resources/stones.jpg')

	stones_small = skimage.transform.resize(stones_image[:,:stones_image.shape[0]], (h, w))
	grass_small  = skimage.transform.resize(grass_image[:,:grass_image.shape[0]], (h, w))

	return grass_small, stones_small


def scale_images(images):
	np.add(images, -.5, out=images)
	np.multiply(images, 4, out=images)


def generate_easy_batch(w=128, h=128, minibatch_size=10, sides=5):
	input_data = np.zeros([minibatch_size, h, w, 1], dtype=np.float32)
	label_data = np.zeros([minibatch_size, h, w, 1], dtype=np.int64)

	for i in range(minibatch_size):
		mask = random_polygon(w, h, sides)
		input_data[i, :, :, 0] = 2.*mask - 1.
		label_data[i, :, :, 0] = mask

	return input_data, label_data


def generate_color_batch(fg, bg, w=128, h=128, minibatch_size=10, sides=5):
	input_data = np.zeros([minibatch_size, h, w, 3], dtype=np.float32)
	label_data = np.zeros([minibatch_size, h, w, 1], dtype=np.int64)

	for i in range(minibatch_size):
		img, mask = random_textured_polygon(fg, bg, w, h, sides)
		input_data[i, :, :] = img
		label_data[i, :, :] = mask

	scale_images(input_data)

	return input_data, label_data


def generate_grayscale_batch(fg, bg, w=128, h=128, minibatch_size=10, sides=5):
	input_data = np.zeros([minibatch_size, h, w, 1], dtype=np.float32)
	label_data = np.zeros([minibatch_size, h, w, 1], dtype=np.int64)

	for i in range(minibatch_size):
		img, mask = random_textured_polygon(fg, bg, w, h, sides)
		img_gray = skimage.color.rgb2gray(img)
		input_data[i, :, :, 0] = img_gray
		label_data[i, :, :] = mask

	scale_images(input_data)

	return input_data, label_data
