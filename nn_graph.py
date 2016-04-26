import tensorflow as tf

input_channels = 1

conv1_size = 5
conv1_stride = 1
conv1_features = 16

output_conv_size = 5
output_stride = 1
output_features = 2

random_init_stddev = 1e-4

def build_graph(input_size, minibatch_size):
	flat_size = minibatch_size * input_size * input_size

	inputs = tf.placeholder(tf.float32, shape=[minibatch_size, input_size, input_size, input_channels], name='inputs')
	labels = tf.placeholder(tf.int64, shape=[minibatch_size, input_size, input_size, 1], name='labels')

	with tf.name_scope('conv1') as scope:
		conv1_init     = tf.truncated_normal([conv1_size, conv1_size, input_channels, conv1_features], stddev=random_init_stddev, dtype=tf.float32)
		conv1_weights  = tf.Variable(conv1_init, name='weights')
		conv1_bias     = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32), name='bias')

		conv1_y        = tf.nn.conv2d(inputs, conv1_weights, [1, conv1_stride, conv1_stride, 1], padding='SAME', name='y')
		conv1_bias     = tf.nn.bias_add(conv1_y, conv1_bias)
		conv1_activity = tf.nn.relu(conv1_bias , name='activity')

	with tf.name_scope('output') as scope:
		output_init    = tf.truncated_normal([output_conv_size, output_conv_size, conv1_features, output_features], stddev=random_init_stddev, dtype=tf.float32)
		output_weights = tf.Variable(output_init, name='weights')
		output_bias    = tf.Variable(tf.zeros([output_features], dtype=tf.float32), name='bias')

		output_y       = tf.nn.conv2d(conv1_activity, output_weights, [1, output_stride, output_stride, 1], padding='SAME', name='y')
		output_logits  = tf.nn.bias_add(output_y, output_bias)

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

	return inputs, labels, output_pr, loss, accuracy, conv1_weights
