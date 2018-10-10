import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from categorical import sample_gumbel, gumbel_softmax_sample, gumbel_softmax


training_data = tf.placeholder(tf.float32, [None, 784])

alpha = tf.Variable(tf.random_normal([784, 25, 784]), name = "alpha")
s_alpha = tf.nn.softmax(alpha)

dist = gumbel_softmax(s_alpha, 0.5, True)
print(dist)


print(tf.gradients(dist, alpha))

um_ok = tf.einsum("abc,dc->dab",dist, training_data)
print(tf.gradients(um_ok, alpha))
print(um_ok)
lmao = tf.argmax(dist, axis=2, output_type=tf.int32)
print(tf.gradients(lmao, alpha))

temp = tf.gather(training_data, lmao, axis = 1)
print(temp)
init = tf.global_variables_initializer()
tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
tf_config.gpu_options.allow_growth = True

with tf.Session() as sess:
    sess.run(init)
    step = 1
    a = sess.run(dist)
    exit(0)
