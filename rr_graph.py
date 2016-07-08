import tensorflow as tf

input_channels = 1

conv1_size = 5
conv1_stride = 2
conv1_features = 2

output_conv_size = 5
output_stride = 1
output_features = 1

random_init_stddev = 1e-4

def build_graph(input_size, minibatch_size):
	flat_size = conv1_features * input_size//2 * input_size//2

	inputs = tf.placeholder(tf.float32, shape=[minibatch_size, input_size, input_size, input_channels], name='inputs')
	labels = tf.placeholder(tf.float32, shape=[minibatch_size], name='labels')

	with tf.name_scope('conv1') as scope:
		conv1_init     = tf.truncated_normal([conv1_size, conv1_size, input_channels, conv1_features], stddev=random_init_stddev, dtype=tf.float32)
		conv1_weights  = tf.Variable(conv1_init, name='weights')
		conv1_bias     = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32), name='bias')

		conv1_y        = tf.nn.conv2d(inputs, conv1_weights, [1, conv1_stride, conv1_stride, 1], padding='SAME', name='y')
		conv1_biased   = tf.nn.bias_add(conv1_y, conv1_bias)
		conv1_activity = tf.tanh(conv1_biased , name='activity')

	with tf.name_scope('output') as scope:
		output_init    = tf.truncated_normal([flat_size, output_features], stddev=random_init_stddev, dtype=tf.float32)
		output_weights = tf.Variable(output_init, name='weights')
		output_bias    = tf.Variable(tf.zeros([output_features], dtype=tf.float32), name='bias')

		conv1_flat     = tf.reshape(conv1_activity, [minibatch_size, flat_size])

		output_y       = tf.matmul(conv1_flat, output_weights, name='y')
		output_raw     = tf.nn.bias_add(output_y, output_bias)
		output_tanh    = tf.tanh(output_raw)
		output         = tf.reshape(output_tanh, [minibatch_size])

	with tf.name_scope('loss') as scope:
		minibatch_loss = tf.squared_difference(labels, output)
		loss = tf.reduce_mean(minibatch_loss)
		tf.scalar_summary('loss', loss)

	return inputs, labels, output, loss
