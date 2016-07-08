# Run "python nn_train.py" first.

import tensorflow as tf
import numpy as np
import skimage.color
import skimage.io

import nn_generate_data
import nn_graph

# Generate random test image

input_size = 128

# fg, bg = nn_generate_data.load_default_textures(w=input_size, h=input_size)
# image_rgb, mask = nn_generate_data.random_textured_polygon(fg, bg, w=input_size, h=input_size)
# mask  = mask[np.newaxis, :, :]

# Use random image found on the internet (cropped to be square)

image_rgb = skimage.io.imread("resources/grass-pebbles-stone-1938394.jpg")
input_size = image_rgb.shape[0]       # assume square image
# skimage.io.imshow(image_rgb)
# skimage.io.show()

# Prepare image

image = skimage.color.rgb2gray(image_rgb)[np.newaxis, :, :, np.newaxis]
nn_generate_data.scale_images(image)

# Load pre-trained TensorFlow graph

inputs, labels, output_pr, loss, accuracy, conv1_weights = nn_graph.build_graph(input_size=input_size, minibatch_size=1)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, '/tmp/nn-detect-log/checkpoint-1000')

feed = {inputs: image}
output_pr_value = sess.run(output_pr, feed_dict = feed)

# Show results

import matplotlib.pyplot as plt

plt.imshow(output_pr_value[0,:,:,0], interpolation='nearest')
plt.colorbar()
plt.figure()
plt.imshow(output_pr_value[0,:,:,1], interpolation='nearest')
plt.colorbar()
plt.show()
