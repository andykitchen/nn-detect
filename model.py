import debug_data

import numpy as np
import tensorflow as tf

minibatch_size = 16
input_size = 128
input_channels = 1

conv1_size = 5
conv1_stride = 1
conv1_features = 8

output_conv_size = 1
output_stride = 1
output_features = 2

num_iterations = 10

# if checkpoint exists, load from checkpoint

np.random.seed(42)

with tf.Session() as sess:
	inputs = tf.placeholder(tf.float32, shape=[minibatch_size, input_size, input_size, input_channels], name='inputs')
	labels = tf.placeholder(tf.int64, shape=[minibatch_size, input_size, input_size, 1], name='labels')

	with tf.name_scope('conv1') as scope:
		conv1_weights  = tf.Variable(0.01*np.random.randn(conv1_size, conv1_size, input_channels, conv1_features).astype(np.float32), name='weights')
		conv1_bias     = tf.Variable(np.zeros([conv1_features], dtype=np.float32), name='bias')

		conv1_y        = tf.nn.conv2d(inputs, conv1_weights, [1, conv1_stride, conv1_stride, 1], padding='SAME', name='y')
		conv1_activity = tf.nn.relu(conv1_y + conv1_bias, name='activity')

	with tf.name_scope('output') as scope:
		output_weights     = tf.Variable(0.01*np.random.randn(output_conv_size, output_conv_size, conv1_features, output_features).astype(np.float32))
		output_bias        = tf.Variable(np.zeros([output_features], dtype=np.float32))

		output_y           = tf.nn.conv2d(conv1_activity, output_weights, [1, output_stride, output_stride, 1], padding='SAME')
		output_logits      = output_y + output_bias
		output_probability = tf.sigmoid(output_logits)

	with tf.name_scope('loss') as scope:
		flat_size = minibatch_size * input_size * input_size

		output_logits_flat = tf.reshape(output_logits, [flat_size, output_features])
		labels_flat = tf.reshape(labels, [flat_size])

		loss_raw = tf.nn.sparse_softmax_cross_entropy_with_logits(output_logits_flat, labels_flat)
		loss = tf.reduce_mean(loss_raw)

	trainable_vars = tf.trainable_variables()

	opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
	opt_op = opt.minimize(loss, var_list=trainable_vars)

	writer = tf.train.SummaryWriter("/tmp/nn-detect-log",
	                                sess.graph.as_graph_def(add_shapes=True))

	init_op = tf.initialize_all_variables()
	init_op.run()

	for i in range(num_iterations):
		input_data, label_data = debug_data.generate_batch(w=input_size, h=input_size, minibatch_size=minibatch_size)
		_, loss_val = sess.run([opt_op, loss], feed_dict={inputs: input_data, labels: label_data})
		print "current loss: ", loss_val
		# ...save current loss
		# if 50 iterations have happened
		#    save checkpoint
