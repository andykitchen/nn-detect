import numpy as np
import tensorflow as tf

import os

image_files = [os.path.join(root, file) \
    for root, dirs, files in os.walk("resources") \
        for file in files \
            if file.endswith(".jpeg") or file.endswith(".jpg")] 

image_file_queue = tf.train.string_input_producer(image_files, num_epochs=3, shuffle=True)
dequeue_image_file = image_file_queue.dequeue()

reader = tf.WholeFileReader()
image_file_name, image_data = reader.read(image_file_queue)
read_image = tf.image.decode_jpeg(image_data)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():
            name, im = sess.run([image_file_name, read_image])
            print name, im.shape

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
