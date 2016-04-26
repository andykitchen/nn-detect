import debug_data

import numpy as np
import tensorflow as tf

minibatch_size = 16

input_size = 128
input_channels = 1

conv1_size = 5
conv1_stride = 1
conv1_features = 16

output_conv_size = 5
output_stride = 1
output_features = 2

num_iterations = 1000
learning_rate = 0.1
report_frequency = 10
checkpoint_frequency = 10
random_init_stddev = 1e-4

validation_frequency = 10
validation_batches = 8

flat_size = minibatch_size * input_size * input_size

random_seed = 42
validation_random_seed = 64

log_path = '/tmp/nn-detect-log'
checkpoint_base_path = log_path + '/checkpoint'


def get_validation_data(fg, bg):
	np.random.seed(validation_random_seed)
	validation_data = []
	for i in range(validation_batches):
		validation_data.append(debug_data.generate_grayscale_batch(fg, bg, w=input_size, h=input_size, minibatch_size=minibatch_size))
	return validation_data


np.random.seed(random_seed)
tf.set_random_seed(random_seed)

with tf.Session() as sess:
	inputs = tf.placeholder(tf.float32, shape=[minibatch_size, input_size, input_size, input_channels], name='inputs')
	labels = tf.placeholder(tf.int64, shape=[minibatch_size, input_size, input_size, 1], name='labels')

	with tf.name_scope('conv1') as scope:
		conv1_init     = tf.truncated_normal([conv1_size, conv1_size, input_channels, conv1_features], stddev=random_init_stddev, dtype=tf.float32)
		conv1_weights  = tf.Variable(conv1_init, name='weights')
		conv1_bias     = tf.Variable(tf.zeros([conv1_features], dtype=np.float32), name='bias')

		conv1_y        = tf.nn.conv2d(inputs, conv1_weights, [1, conv1_stride, conv1_stride, 1], padding='SAME', name='y')
		conv1_bias     = tf.nn.bias_add(conv1_y, conv1_bias)
		conv1_activity = tf.nn.relu(conv1_bias , name='activity')

	with tf.name_scope('output') as scope:
		output_init    = tf.truncated_normal([output_conv_size, output_conv_size, conv1_features, output_features], stddev=random_init_stddev, dtype=tf.float32)
		output_weights = tf.Variable(output_init, name='weights')
		output_bias    = tf.Variable(tf.zeros([output_features], dtype=np.float32), name='bias')

		output_y       = tf.nn.conv2d(conv1_activity, output_weights, [1, output_stride, output_stride, 1], padding='SAME', name='y')
		output_logits  = output_y + output_bias

		output_logits_flat = tf.reshape(output_logits, [flat_size, output_features])
		output_pr_flat = tf.nn.softmax(output_logits_flat)
		output_pr      = tf.reshape(output_pr_flat, [minibatch_size, input_size, input_size, output_features])

	with tf.name_scope('loss') as scope:
		output_logits_flat = tf.reshape(output_logits, [flat_size, output_features])
		labels_flat = tf.reshape(labels, [flat_size])

		loss_raw = tf.nn.sparse_softmax_cross_entropy_with_logits(output_logits_flat, labels_flat, name='xentropy')
		loss = tf.reduce_mean(loss_raw)
		tf.scalar_summary('loss', loss)

	with tf.name_scope('accuracy') as scope:
		predictions = tf.argmax(output_pr, dimension=3)
		predictions_reshaped = tf.reshape(predictions, [minibatch_size, input_size, input_size, 1])
		correct_predictions = tf.to_float(tf.equal(labels, predictions_reshaped))
		accuracy = tf.reduce_mean(correct_predictions)


	trainable_vars = tf.trainable_variables()

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss, var_list=trainable_vars)

	writer = tf.train.SummaryWriter(log_path, sess.graph.as_graph_def(add_shapes=True))

	saver = tf.train.Saver(max_to_keep=0)

	latest_checkpoint = tf.train.latest_checkpoint(log_path)

	if latest_checkpoint is not None:
		print "restoring model from checkpoint:", latest_checkpoint
		saver.restore(sess, latest_checkpoint)
		starting_iteration = int(latest_checkpoint.split('-')[-1]) + 1
	else:
		print "initializing new model..."
		tf.initialize_all_variables().run()
		starting_iteration = 0

	summary_op = tf.merge_all_summaries()

	fg, bg = debug_data.load_default_textures(input_size, input_size)
	validation_data = get_validation_data(fg, bg)

	for i in xrange(starting_iteration, num_iterations):
		input_data, label_data = debug_data.generate_grayscale_batch(fg, bg, w=input_size, h=input_size, minibatch_size=minibatch_size)
		feed = {inputs: input_data, labels: label_data}
		_, summary_value, loss_value = sess.run([train_op, summary_op, loss], feed_dict=feed)


		writer.add_summary(summary_value, i)

		if i % report_frequency == 0:
			print "iteration:", i, "loss:", loss_value

		if i % validation_frequency == 0:
			accuracy_values = []
			for validation_input_data, validation_label_data in validation_data:
				feed = {inputs: validation_input_data, labels: validation_label_data}
				accuracy_value = sess.run(accuracy, feed_dict=feed)
				accuracy_values.append(accuracy_value)

			print "iteration:", i, "validation accuracy:", np.array(accuracy_value).mean()

		if i % checkpoint_frequency == 0:
			checkpoint_path = saver.save(sess, checkpoint_base_path, global_step=i)
			print "checkpointed:", checkpoint_path
