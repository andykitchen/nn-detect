import numpy as np
import tensorflow as tf

import os

image_files = [os.path.join(root, file) \
    for root, dirs, files in os.walk("images") \
        for file in files \
            if file.startswith("image_")] 

mask_files = [file.replace("image_", "mask_") for file in image_files]

q = tf.FIFOQueue(capacity=3, dtypes=[tf.string, tf.string])
enq = q.enqueue_many([image_files, mask_files])
qr = tf.train.QueueRunner(q, [enq])
tf.train.add_queue_runner(qr)

image_path, mask_path = q.dequeue()

reader = tf.WholeFileReader()
image_file_name, image_data = reader.read(q)
read_image = tf.image.decode_jpeg(image_data)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():
            n1, n2 = sess.run([image_path, mask_path])
            print n1, n2

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
