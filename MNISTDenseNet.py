import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from categorical import sample_gumbel, gumbel_softmax_sample, gumbel_softmax
from tensorflow.examples.tutorials.mnist import input_data
from Dataset import DataSet
import seaborn as sns
from DenseLayer import DenseLayer

# Bilevel optimization from DARTS
debug = True
mnist = input_data.read_data_sets("data/mluo/tmp/data/", one_hot=True)

batch_size = 512
num_iterations = 1000000

# Input Data and Labels
training_data = tf.placeholder(tf.float32, [None, 784])
training_labels = tf.placeholder(tf.float32, [None, 10])
train_weights = tf.placeholder(tf.int32, name = "train_weights")

# Defining Variables
with tf.variable_scope("Weights", reuse = tf.AUTO_REUSE):
  layer3 = tf.Variable(tf.random_normal([14*14*32, 1024]))
  bias3 = tf.Variable(tf.random_normal([1024]))
  layer4 = tf.Variable(tf.random_normal([1024, 10]))
  bias4 = tf.Variable(tf.random_normal([10]))

# Building Graph

#-----------------------------------First Layer-----------------------------------#
'''
s_alpha = tf.nn.softmax(alpha)

dist = gumbel_softmax(s_alpha, 0.5, True)
temp = tf.einsum("abc,dc->dab",dist, training_data)

#dist = tf.argmax(dist, axis=2, output_type=tf.int32)
gg = tf.cond(train_weights > 0, lambda: tf.gather(training_data, tf.argmax(s_alpha, axis = 2, output_type=tf.int32), axis=1), lambda: temp)
#temp = tf.gather(training_data, gg, axis = 1)
print(gg)
output = tf.reshape(tf.squeeze(tf.einsum("aij,bjk->abik",gg, conv_weights)), [batch_size, 28, 28, 32])
print(output)
x = tf.nn.bias_add(output, bias1) 
x = tf.nn.relu(x)
'''
x = DenseLayer(training_data, output = 784, num_weights_per_filter = 25, num_filters = 32, weight_train = train_weights)
x = tf.reshape(x, [batch_size, 28, 28, 32])   
#1st Pooling Layer
x = tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

hyper_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Hyperparameters')
weight_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Weights')

#----------------------------------------------------------------------------------#

#Fully-Connected Layer
x = tf.reshape(x, [-1, 14*14*32])
x = tf.add(tf.matmul(x, layer3), bias3)
x = tf.nn.relu(x) 

output =  tf.add(tf.matmul(x, layer4), bias4)

# Calculating Loss Function
loss_vector = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=training_labels)
cost = tf.reduce_mean(loss_vector)

with tf.variable_scope("Optimizer"):
  optimizer_weight = tf.train.AdamOptimizer(0.001)
  optimizer_hyper = tf.train.AdamOptimizer(0.001)

weight_grad = optimizer_weight.compute_gradients(cost, weight_vars) 

hyper_grad = optimizer_hyper.compute_gradients(cost, hyper_vars) 

optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "Optimizer")

update = tf.cond(train_weights>0, lambda: optimizer_weight.apply_gradients(weight_grad), lambda: optimizer_hyper.apply_gradients(hyper_grad))
correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(training_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#---------------------------------------------------------------------------------------------------------------------------#

init = tf.global_variables_initializer()
tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
tf_config.gpu_options.allow_growth = True

#hyper_train = DataSet(mnist.test.images[:5000], mnist.test.labels[:5000])
#validation = DataSet(mnist.test.images[5000:], mnist.test.labels[5000:])

x = np.arange(784)

with tf.Session() as sess:
    sess.run(init)
    meta_step = 0

    # Meta-iterations
    while meta_step <300:
      
      print("----------------------------META-ITERATION " + str(meta_step) + "----------------------------")
      sess.run(tf.initialize_variables(optimizer_scope))
      print("SWITCHING TO WEIGHT OPT")
      # Weight optimization
      weight_step = 0
      while(weight_step<1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        opt = sess.run(update, feed_dict={training_data: batch_x, training_labels: batch_y, train_weights: 1})

        if(weight_step%1==0):
          loss, acc = sess.run([cost, accuracy], feed_dict={training_data: batch_x,
                                                           training_labels: batch_y, train_weights: 1})
          print("Weight Iter " + str(weight_step) + " with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))
        weight_step+=1

      batch_x, batch_y = mnist.test.next_batch(batch_size)
      loss, acc = sess.run([cost, accuracy], feed_dict={training_data: batch_x,
                                                           training_labels: batch_y, train_weights: 1})
      print("Metaiteration Step " + str(meta_step) + " with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Validation Accuracy= " +  "{:.5f}".format(acc))


      print("SWITCHING TO HYPERPARAM OPT")
      #Hyperparameter optimization
      sess.run(tf.initialize_variables(optimizer_scope))
      hyper_step =0
      while(hyper_step<1):
        batch_x, batch_y = mnist.test.next_batch(batch_size)
        opt = sess.run(update, feed_dict={training_data: batch_x, training_labels: batch_y, train_weights: 0})



        if(hyper_step%1==0):
          loss, acc = sess.run([cost, accuracy], feed_dict={training_data: batch_x,
                                                           training_labels: batch_y, train_weights: 1})
          print("Hyper Iter " + str(hyper_step) + " with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))
        hyper_step+=1

      if(meta_step%10==0):
        a = sess.run(hyper_vars[0])
        
        #sns.barplot(x, y = a[0][0])
        print(np.max(a, axis=2))
        print(np.argmax(a, axis=2))
      meta_step+=1
    
    #Fine tune network now
    weight_step = 0
    sess.run(tf.initialize_variables(weight_vars))
    while(weight_step<6000):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        opt = sess.run(update, feed_dict={training_data: batch_x, training_labels: batch_y, train_weights: 1})

        if(weight_step%100==0):
          loss, acc = sess.run([cost, accuracy], feed_dict={training_data: batch_x,
                                                           training_labels: batch_y, train_weights: 1})
          print("Weight Iter " + str(weight_step) + " with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))
          acc = sess.run([accuracy], feed_dict={training_data: mnist.test.images[:batch_size],
                                      training_labels: mnist.test.labels[:batch_size], train_weights: 1})
          print("Accuracy MNIST:", acc)

        weight_step+=1


    acc = sess.run([accuracy], feed_dict={training_data: mnist.test.images[:batch_size],
                                      training_labels: mnist.test.labels[:batch_size], train_weights: 1})
    print("Accuracy MNIST:", acc)
