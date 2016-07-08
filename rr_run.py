# Run "python rr_train.py" first.

import tensorflow as tf
import numpy as np
import skimage.color
import skimage.io

import rr_generate_data
import rr_graph

# Generate random test image

input_size = 128

image, mask = rr_generate_data.random_road(height=input_size, width=input_size)
rr_generate_data.normal_distribution(image.astype(np.float32))

# Load pre-trained TensorFlow graph

inputs, labels, output_pr, loss = rr_graph.build_graph(input_size=input_size, minibatch_size=1)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, '/tmp/rr-detect-log/checkpoint-300')

feed = {inputs: image[np.newaxis, :, :, :]}
output_pr_value = sess.run(output_pr, feed_dict = feed)

# Show results

import matplotlib.pyplot as plt

print(output_pr_value)
plt.imshow(image[:,:,0], interpolation='nearest')
plt.colorbar()
plt.show()
