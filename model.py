import numpy as np
import tensorflow as tf

import nn_generate_data
import nn_graph

input_size = 128
minibatch_size = 16

num_iterations = 1000
learning_rate = 0.1
report_frequency = 10
checkpoint_frequency = 10

validation_frequency = 10
validation_batches = 8

random_seed = 42
validation_random_seed = 64

log_path = '/tmp/nn-detect-log'
checkpoint_base_path = log_path + '/checkpoint'


def get_validation_data(fg, bg):
	np.random.seed(validation_random_seed)
	validation_data = []
	for i in range(validation_batches):
		validation_data.append(nn_generate_data.generate_grayscale_batch(fg, bg, w=input_size, h=input_size, minibatch_size=minibatch_size))
	return validation_data


np.random.seed(random_seed)
tf.set_random_seed(random_seed)

with tf.Session() as sess:
	inputs, labels, output_pr, loss, accuracy, conv1_weights = nn_graph.build_graph(input_size, minibatch_size)

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

	fg, bg = nn_generate_data.load_default_textures(input_size, input_size)
	validation_data = get_validation_data(fg, bg)

	for i in xrange(starting_iteration, num_iterations + 1):
		input_data, label_data = nn_generate_data.generate_grayscale_batch(fg, bg, w=input_size, h=input_size, minibatch_size=minibatch_size)
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
