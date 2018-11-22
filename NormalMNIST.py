import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

print("hi")
# Input Data and Labels
training_data = tf.placeholder(tf.float32, [None, 784])
training_labels = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# Defining Variables
layer1 = tf.Variable(tf.random_normal([5,5,1,32]))
bias1 = tf.Variable(tf.random_normal([32]))

# 1st Convoluted Layer
x = tf.reshape(training_data, shape = [-1, 28, 28, 1])
hi1 = tf.nn.conv2d(x, layer1, strides = [1,1,1,1], padding = 'SAME')
x = tf.nn.bias_add(hi1, bias1)
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
x = tf.nn.dropout(x, keep_prob)

#Converting to class scores
layer4 = tf.Variable(tf.random_normal([1024, 10]))
bias4 = tf.Variable(tf.random_normal([10]))

output =  tf.add(tf.matmul(x, layer4), bias4)

# Calculating Loss Function
loss_vector = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=training_labels)
cost = tf.reduce_mean(loss_vector)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(training_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



#---------------------------------------------------------------------------------------------------------------------------#

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    
    # Keep training until reach max iterations
    while step <4000:
        batch_x, batch_y = mnist.train.next_batch(128)
        wasd, opt = sess.run([hi1, optimizer], feed_dict = {training_data: batch_x, training_labels: batch_y, keep_prob: 0.75})
        if(step%100==0):
        	loss, acc = sess.run([cost, accuracy], feed_dict={training_data: batch_x,
                                                           training_labels: batch_y, keep_prob: 1
                                                              })
        	print("Iter " + str(step) + " with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))
        step += 1
    for i in range(0, 10):
        batch_x, batch_y = mnist.test.next_batch(512)
        acc = sess.run([accuracy], feed_dict={training_data: batch_x,
                                      training_labels: batch_y,
                                      keep_prob: 1.})
        print("Accuracy MNIST:", acc)
    saver = tf.train.Saver(tf.trainable_variables())
    save_path = saver.save(sess, "../model/model.ckpt")