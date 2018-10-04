import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from categorical import sample_gumbel, gumbel_softmax_sample, gumbel_softmax
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/mluo/tmp/data/", one_hot=True)

batch_size = 512

# Input Data and Labels
training_data = tf.placeholder(tf.float32, [None, 784])
training_labels = tf.placeholder(tf.float32, [None, 10])
condition = tf.placeholder(tf.int32, name = "condition")
# Defining Variables
conv_weights = tf.Variable(tf.random_normal([32, 25, 1]))

alpha = tf.Variable(tf.random_normal([784, 25, 784]), name = "alpha")
s_alpha = tf.nn.softmax(alpha)

dist = gumbel_softmax(s_alpha, 0.5, True)
dist = tf.argmax(s_alpha, axis=2, output_type=tf.int32)

#  tf.contrib.distributions.Categorical(probs = s_alpha).sample(1)[0]
gg = tf.cond(condition > 0, lambda: dist, lambda: tf.argmax(s_alpha, axis = 2, output_type=tf.int32))
temp = tf.gather(training_data, gg, axis = 1)

output = tf.reshape(tf.squeeze(tf.einsum("aij,bjk->abik",temp, conv_weights)), [batch_size, 28, 28, 32])

bias1 = tf.Variable(tf.random_normal([32]))

x = tf.nn.bias_add(output, bias1)
x = tf.nn.relu(x)
#1st Pooling Layer
x = tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

#Defining Variables
layer3 = tf.Variable(tf.random_normal([14*14*32, 1024]))
bias3 = tf.Variable(tf.random_normal([1024]))

#Fully-Connected Layer
x = tf.reshape(x, [-1, 14*14*32])
x = tf.add(tf.matmul(x, layer3), bias3)
x = tf.nn.relu(x)

#Converting to class scores
layer4 = tf.Variable(tf.random_normal([1024, 10]))
bias4 = tf.Variable(tf.random_normal([10]))

output =  tf.add(tf.matmul(x, layer4), bias4)

# Calculating Loss Function
loss_vector = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=training_labels)
cost = tf.reduce_mean(loss_vector)
o = tf.train.AdamOptimizer(learning_rate=0.05)
optimizer = o.minimize(cost)

gradients = o.compute_gradients(cost)
for i, (a,b) in enumerate(gradients):
	print(a)


correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(training_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#---------------------------------------------------------------------------------------------------------------------------#

init = tf.global_variables_initializer()
tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
tf_config.gpu_options.allow_growth = True
a = 0
b = 1
with tf.Session() as sess:
    sess.run(init)
    step = 1

    # Keep training until reach max iterations
    while step <10000:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        if(step<5000):
                d, opt = sess.run([dist,optimizer], feed_dict = {training_data: batch_x, training_labels: batch_y, condition: b})
                print(d)
                print(d.shape)
                exit(0)
        else:
                opt = sess.run(optimizer, feed_dict = {training_data: batch_x, training_labels: batch_y, condition: a})
        if(step%100==0):
          lol, loss, acc = sess.run([gg, cost, accuracy], feed_dict={training_data: batch_x,
                                                           training_labels: batch_y, condition: a})
          print(lol[0])
          print("Iter " + str(step) + " with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))
        step += 1


    acc = sess.run([accuracy], feed_dict={training_data: mnist.test.images[:batch_size],
                                      training_labels: mnist.test.labels[:batch_size], condition: a})
    print("Accuracy MNIST:", acc)
