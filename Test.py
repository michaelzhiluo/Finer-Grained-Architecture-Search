import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from categorical import sample_gumbel, gumbel_softmax_sample, gumbel_softmax
from tensorflow.examples.tutorials.mnist import input_data

# Bilevel optimization from DARTS

mnist = input_data.read_data_sets("data/mluo/tmp/data/", one_hot=True)

batch_size = 512
num_iterations = 1000000

# Input Data and Labels
training_data = tf.placeholder(tf.float32, [None, 784])
training_labels = tf.placeholder(tf.float32, [None, 10])
condition = tf.placeholder(tf.int32, name = "condition")

# Defining Variables
with tf.variable_scope("Weights"):
  conv_weights = tf.Variable(tf.random_normal([32, 25, 1]))
  bias1 = tf.Variable(tf.random_normal([32]))
  layer3 = tf.Variable(tf.random_normal([14*14*32, 1024]))
  bias3 = tf.Variable(tf.random_normal([1024]))
  layer4 = tf.Variable(tf.random_normal([1024, 10]))
  bias4 = tf.Variable(tf.random_normal([10]))

weight_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Weights')

with tf.variable_scope("Hyperparameters"):
  alpha = tf.Variable(tf.random_normal([784, 25, 784]), name = "alpha")

hyper_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Hyperparameters')

# Building Graph

s_alpha = tf.nn.softmax(alpha)

dist = gumbel_softmax(s_alpha, 0.5, True)
temp = tf.einsum("abc,dc->dab",dist, training_data)

#dist = tf.argmax(dist, axis=2, output_type=tf.int32)
gg = tf.cond(condition > 0, lambda: tf.gather(training_data, tf.argmax(s_alpha, axis = 2, output_type=tf.int32), axis=1), lambda: temp)
#temp = tf.gather(training_data, gg, axis = 1)

output = tf.reshape(tf.squeeze(tf.einsum("aij,bjk->abik",gg, conv_weights)), [batch_size, 28, 28, 32])

x = tf.nn.bias_add(output, bias1)
x = tf.nn.relu(x)

#1st Pooling Layer
x = tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


#Fully-Connected Layer
x = tf.reshape(x, [-1, 14*14*32])
x = tf.add(tf.matmul(x, layer3), bias3)
x = tf.nn.relu(x)

output =  tf.add(tf.matmul(x, layer4), bias4)

# Calculating Loss Function
loss_vector = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=training_labels)
cost = tf.reduce_mean(loss_vector)

with tf.variable_scope("Optimizer"):
  optimizer_weight = tf.train.AdamOptimizer(0.05)
weight_grad = optimizer_weight.compute_gradients(cost, weight_vars)

optimizer_hyper = tf.train.AdamOptimizer(0.5)
hyper_grad = optimizer_hyper.compute_gradients(cost, hyper_vars)  

optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "Optimizer")

update = tf.cond(condition>0, lambda: optimizer_weight.apply_gradients(weight_grad), lambda: optimizer_hyper.apply_gradients(hyper_grad))

correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(training_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#---------------------------------------------------------------------------------------------------------------------------#

init = tf.global_variables_initializer()
tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
tf_config.gpu_options.allow_growth = True


with tf.Session() as sess:
    sess.run(init)
    meta_step = 1

    # Meta-iterations
    while meta_step <1000:
      print("----------------------------META-ITERATION " + str(meta_step) + "----------------------------")
      print("SWITCHING TO WEIGHT OPT")
      # Weight optimization
      weight_step = 0
      sess.run(tf.initialize_variables(optimizer_scope))
      while(weight_step<100):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        opt = sess.run(update, feed_dict={training_data: batch_x, training_labels: batch_y, condition: 1})

        if(weight_step%10==0):
          loss, acc = sess.run([cost, accuracy], feed_dict={training_data: batch_x,
                                                           training_labels: batch_y, condition: 1})
          print("Weight Iter " + str(weight_step) + " with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))
        weight_step+=1

      print("SWITCHING TO HYPERPARAM OPT")
      #Hyperparam optimization
      hyper_step =0
      while(hyper_step<3):
        batch_x, batch_y = mnist.test.next_batch(batch_size)
        opt = sess.run(update, feed_dict={training_data: batch_x, training_labels: batch_y, condition: 0})

        if(hyper_step%1==0):
          loss, acc = sess.run([cost, accuracy], feed_dict={training_data: batch_x,
                                                           training_labels: batch_y, condition: 1})
          print("Hyper Iter " + str(hyper_step) + " with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))
        hyper_step+=1
      meta_step+=1

    acc = sess.run([accuracy], feed_dict={training_data: mnist.test.images[:batch_size],
                                      training_labels: mnist.test.labels[:batch_size], condition: 1})
    print("Accuracy MNIST:", acc)
