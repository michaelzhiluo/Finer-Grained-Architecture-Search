import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from categorical import sample_gumbel, gumbel_softmax_sample, gumbel_softmax
from tensorflow.examples.tutorials.mnist import input_data
from Dataset import DataSet
import seaborn as sns
from utils import DenseLayer
from tensorflow.python.tools import inspect_checkpoint as chkp
from IPython import embed
from tensorflow.python import pywrap_tensorflow
import os
import logging
import sys
from tensorflow.examples.tutorials.mnist import input_data

class GeneralizedConvNetwork(object):
	def __init__(self):
		print("HI")
		self.batch_size = 256
		self.weight_lr = 0.001
		self.meta_lr = 0.01
		self.beta = 0.5
		self.build_model()
		return


	def build_model(self):
		#Placeholders
		self.train_input = tf.placeholder(tf.float32, [None, 784])
		self.train_label = tf.placeholder(tf.float32, [None, 10])
		self.test_input = tf.placeholder(tf.float32, [None, 784])
		self.test_label = tf.placeholder(tf.float32, [None, 10])

		#Additional Placeholders
		self.train_weights = tf.placeholder(tf.int32, name = "train_weights")
		self.exploration = tf.placeholder(tf.int32, name = "exploration")

		#Weights
		self.weights = {}
		self.kernel_size = 5
		self.num_filters = 32
		with tf.variable_scope("Weights", reuse = tf.AUTO_REUSE):
			self.weights['w0'] = tf.Variable(tf.random_normal([self.num_filters, self.kernel_size*self.kernel_size]))
			self.weights['b0'] = tf.Variable(tf.random_normal([self.num_filters]))
			self.weights['w1'] = tf.Variable(tf.random_normal([14*14*32, 1024]))
			self.weights['b1'] = tf.Variable(tf.random_normal([1024]))
			self.weights['w2'] = tf.Variable(tf.random_normal([1024, 10]))
			self.weights['b2'] = tf.Variable(tf.random_normal([10]))

		self.output = self.feed_forward(self.train_input, self.weights)

		self.train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.train_label))


		grads = tf.gradients(self.train_loss, list(self.weights.values()))
		self.gradients = dict(zip(self.weights.keys(), grads))
		look_ahead_weights = dict(zip(self.weights.keys(), [self.weights[key] - self.beta*self.gradients[key] for key in self.weights.keys()]))

		self.output1 = self.feed_forward(self.test_input, look_ahead_weights)
		self.test_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output1, labels=self.test_label))

		# Optimizer
		with tf.variable_scope("Optimizer"):
			self.weight_optimizer= tf.train.AdamOptimizer(self.weight_lr)
			self.hyper_optimizer= tf.train.AdamOptimizer(self.meta_lr)
		optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "Optimizer")
		self.opt_init = tf.initialize_variables(optimizer_scope)
		# Weight Update Step
		weight_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Weights')
		weight_grad = self.weight_optimizer.compute_gradients(self.train_loss, weight_vars) 
		self.weight_update = self.weight_optimizer.apply_gradients(weight_grad)

		#HyperParameter (Alpha) Update Step
		hyper_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Hyperparameters')
		print(hyper_vars)
		print(weight_vars)
		hyper_grad = self.hyper_optimizer.compute_gradients(self.test_loss, hyper_vars) 
		self.hyper_update = self.hyper_optimizer.apply_gradients(hyper_grad)

		#Statistics
		correct_pred = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.train_label, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	
	def feed_forward(self, inp, weights):
		self.layer1 =  tf.reshape(DenseLayer(inp, 784, weights['w0'], weights['b0'], self.train_weights, self.exploration), [self.batch_size, 28, 28, self.num_filters])
		self.pool1 = tf.reshape(tf.nn.max_pool(self.layer1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME'), [-1, 14*14*self.num_filters])

		self.layer2 = tf.nn.relu(tf.add(tf.matmul(self.pool1, weights['w1']), weights['b1']))

		self.layer3 = tf.nn.relu(tf.add(tf.matmul(self.layer2, weights['w2']), weights['b2']))

		return self.layer3

mnist = input_data.read_data_sets("/data/mluo/tmp/data/", one_hot=True)
sess = tf.Session()

network = GeneralizedConvNetwork()
init = tf.global_variables_initializer()
sess.run(init)
tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
tf_config.gpu_options.allow_growth = True

meta_iterations = 100000
meta_step =0
while meta_step <meta_iterations:


	print("----------------------------META-ITERATION " + str(meta_step) + "----------------------------")
	sess.run(network.opt_init)

	print("SWITCHING TO WEIGHT OPT")
	weight_step = 0
	while(weight_step<1):
		batch_x, batch_y = mnist.train.next_batch(256)
		opt = sess.run(network.weight_update, feed_dict={network.train_input: batch_x, network.train_label: batch_y, network.train_weights: 1, network.exploration: 0})

		if(weight_step%1==0):
			loss, acc = sess.run([network.train_loss, network.accuracy], feed_dict={network.train_input: batch_x, network.train_label: batch_y, network.train_weights: 1, network.exploration: 0})
			print("Weight Iter " + str(weight_step) + " with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))
		weight_step+=1

	print("SWITCHING TO HYPERPARAM OPT")
	hyper_step =0
	while(hyper_step<1):
		train_input, train_label = mnist.train.next_batch(256)
		test_input, test_label = mnist.test.next_batch(256)

		opt = sess.run([network.hyper_update], feed_dict={
			network.train_input: train_input,
			network.train_label: train_label,
			network.test_input: test_input,
			network.test_label: test_label,
			network.exploration: 0,
			network.train_weights: 0
			})
		if(hyper_step%1==0):
			loss, acc = sess.run([network.train_loss, network.accuracy], 
				feed_dict={network.train_input: batch_x, network.train_label: batch_y, network.train_weights: 1, network.exploration: 0})
			print("Hyper Iter " + str(hyper_step) + " with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))
		hyper_step+=1
	
	meta_step+=1