import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from categorical import sample_gumbel, gumbel_softmax_sample, gumbel_softmax
from tensorflow.examples.tutorials.mnist import input_data
from Dataset import DataSet
import seaborn as sns
from DenseLayer import DenseLayer
from tensorflow.python.tools import inspect_checkpoint as chkp
from IPython import embed
from tensorflow.python import pywrap_tensorflow
import os
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Bilevel optimization from DARTS
debug = False
mnist = input_data.read_data_sets("data/mluo/tmp/data/", one_hot=True)

pretrained_weights = False
batch_size = 512
meta_iterations = 100000
exp =0.9
exp_decay = 0.999

# Input Data and Labels
training_data = tf.placeholder(tf.float32, [None, 784])
training_labels = tf.placeholder(tf.float32, [None, 10])
train_weights = tf.placeholder(tf.int32, name = "train_weights")
exploration = tf.placeholder(tf.int32, name = "exploration")

# Defining Variables
with tf.variable_scope("Weights", reuse = tf.AUTO_REUSE):
  layer3 = tf.Variable(tf.random_normal([14*14*32, 1024]))
  bias3 = tf.Variable(tf.random_normal([1024]))
  layer4 = tf.Variable(tf.random_normal([1024, 10]))
  bias4 = tf.Variable(tf.random_normal([10]))


# Building Graph
#-----------------------------------First Layer-----------------------------------#

s_alpha, sampled_connections, x = DenseLayer(training_data, output = 784, num_weights_per_filter = 25, num_filters = 32, weight_train = train_weights, exploration = exploration)
x = tf.reshape(x, [batch_size, 28, 28, 32])   
#1st Pooling Layer
x = tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

hyper_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Hyperparameters')
weight_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Weights')

#For DARTS, where you reinitalize network weights to previous iteration
assign =[]
prev_weight_ph = []
for counter, var in enumerate(weight_vars):
  prev_weight_ph += [tf.placeholder(tf.float32, var.get_shape())]
  assign+=[tf.assign(var, prev_weight_ph[counter])]
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
  optimizer_hyper = tf.train.AdamOptimizer(0.01)

optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "Optimizer")
opt_init = tf.initialize_variables(optimizer_scope)

weight_grad = optimizer_weight.compute_gradients(cost, weight_vars) 

hyper_grad = optimizer_hyper.compute_gradients(cost, hyper_vars) 

loss_placeholder = tf.placeholder(tf.float32)
test_grad = optimzer_hyper.compute_gradients(loss_placeholder, hyper_vars)

update = tf.cond(train_weights>0, lambda: optimizer_weight.apply_gradients(weight_grad), lambda: optimizer_hyper.apply_gradients(hyper_grad))
correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(training_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#---------------------------------------------------------------------------------------------------------------------------#

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
tf_config.gpu_options.allow_growth = True
exit(0)
if pretrained_weights:
  reader = pywrap_tensorflow.NewCheckpointReader("../model/model.ckpt")
  var_to_shape_map = reader.get_variable_to_shape_map()

  restored_weights = []
  for key in var_to_shape_map:
    print("Saved/Restored Tensor_name: ", key)
    temp = reader.get_tensor(key)
    if(temp.shape==(5,5,1,32)):
      temp = temp.reshape((32, 25))
    restored_weights.append(temp)

  for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Weights'):
    if "Adam" not in i.name:
      print(i)
      for j in restored_weights:
        if i.shape==j.shape:
          print(i.shape)
          sess.run(tf.assign(i, tf.convert_to_tensor(j)))
          assert np.array_equal(sess.run(i), j)

#hyper_train = DataSet(mnist.test.images[:5000], mnist.test.labels[:5000])
#validation = DataSet(mnist.test.images[5000:], mnist.test.labels[5000:])

x = np.arange(784)
max_accuracy = 0

meta_step = 0

# Meta-iterations
while meta_step <meta_iterations:

  if(False and meta_step%10==0):
        #a = sess.run(hyper_vars[0])
        #print(a.shape)
        #print(np.max(a, axis=2))
        #print(np.argmax(a, axis=2))
        #print([n.name for n in tf.get_default_graph().as_graph_def().node])
        #print(accuracy)
        #print(sess.run(tf.get_default_graph().get_operation_by_name("softmax_alpha").values())[0].shape)
        '''
        maximum_indexes = sess.run(tf.get_default_graph().get_operation_by_name("argmax_alpha").values())[0]
        hi = np.zeros(784)
        for i in maximum_indexes[0]:
          hi[i] = 1
        fig, ax = plt.subplots()
        sns.heatmap(hi.reshape((28,28)), ax = ax, square=True, cmap="BuPu", cbar = False)  
        fig.savefig("../argmax/" + str(meta_step) + ".png", dpi=96)
        '''
        #embed()
        
        kernels = sess.run(s_alpha)
        
        fig, ax = plt.subplots(5, 5, sharex=True, sharey=True)
        for i, ax in enumerate(ax.flat):
          sns.heatmap(kernels[0][i].reshape((28,28)), ax = ax, square=True, cmap="BuPu", cbar = True)
        fig.savefig("../figures/" + str(meta_step) + ".png", dpi=96, bbox_inches='tight')
        

        #Plots the argmax of weight kernels

        plt.close()
  print("----------------------------META-ITERATION " + str(meta_step) + "----------------------------")
  sess.run(opt_init)
  '''
  DART Update: 
      0) Update w_(t-1) over training set
      1) Save w_(t-1)
      2) Update w_t from w_(t-1) over training set
      3) Using w_t, calculate the loss function
      update alpha_(t-1) to alpha_t over validation set
      4) Reload w_(t-1)
  '''

  print("SWITCHING TO WEIGHT OPT")
  #Step 0: Weight Optimization
  weight_step = 0
  while(weight_step<1):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    opt = sess.run(update, feed_dict={training_data: batch_x, training_labels: batch_y, train_weights: 1, exploration: 1})

    if(weight_step%1==0):
      loss, acc = sess.run([cost, accuracy], feed_dict={training_data: batch_x,
                                                       training_labels: batch_y, train_weights: 1, exploration: 1})
      print("Weight Iter " + str(weight_step) + " with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))
    weight_step+=1

  batch_x, batch_y = mnist.test.next_batch(batch_size)
  loss, acc = sess.run([cost, accuracy], feed_dict={training_data: batch_x,
                                                       training_labels: batch_y, train_weights: 1, exploration: 1})
  
  print("Metaiteration Step " + str(meta_step) + " with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Validation Accuracy= " +  "{:.5f}".format(acc))
  if(max_accuracy < acc):
    max_accuracy = acc
  print("Max Accuracy " + str(max_accuracy))
  
  
  #Step 1
  weight_list = sess.run(weight_vars)

  #Step 2
  batch_x, batch_y = mnist.train.next_batch(batch_size)
  opt = sess.run(update, feed_dict={training_data: batch_x, training_labels: batch_y, train_weights: 1, exploration: 0})  
  

  #Step 3: Hyperparameter optimization
  print("SWITCHING TO HYPERPARAM OPT")
  hyper_step =0
  while(hyper_step<1):
    batch_x, batch_y = mnist.test.next_batch(batch_size)
    if(np.random.sample() <= exp):
      print("EXPLORING")
      opt = sess.run(update, feed_dict={training_data: batch_x, training_labels: batch_y, train_weights: 0, exploration: 1})
    else:
      opt = sess.run(update, feed_dict={training_data: batch_x, training_labels: batch_y, train_weights: 0, exploration: 0})
    exp*=exp_decay
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    if(hyper_step%1==0):
      loss, acc = sess.run([cost, accuracy], feed_dict={training_data: batch_x,
                                                       training_labels: batch_y, train_weights: 1, exploration: 0})
      print("Hyper Iter " + str(hyper_step) + " with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))
    hyper_step+=1

  
  #Step 4: Restore Previous Weights
  for counter, var in enumerate(weight_vars):
    sess.run(assign[counter], feed_dict={prev_weight_ph[counter]: weight_list[counter]})

  assert np.array_equal(weight_list[0], sess.run(weight_vars)[0])
  
  
  meta_step+=1

'''
#Fine tune network now
weight_step = 0
sess.run(tf.initialize_variables(weight_vars))
while(weight_step<10000):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    opt = sess.run(update, feed_dict={training_data: batch_x, training_labels: batch_y, train_weights: 1, exploration: 0})

    if(weight_step%100==0):
      loss, acc = sess.run([cost, accuracy], feed_dict={training_data: batch_x,
                                                       training_labels: batch_y, train_weights: 1, exploration: 0})
      print("Weight Iter " + str(weight_step) + " with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))
      acc = sess.run([accuracy], feed_dict={training_data: mnist.test.images[:batch_size],
                                  training_labels: mnist.test.labels[:batch_size], train_weights: 1, exploration: 0})
      print("Accuracy MNIST:", acc)

    weight_step+=1


acc = sess.run([accuracy], feed_dict={training_data: mnist.test.images[:batch_size],
                                  training_labels: mnist.test.labels[:batch_size], train_weights: 1, exploration: 0})
print("Accuracy MNIST:", acc)
'''
