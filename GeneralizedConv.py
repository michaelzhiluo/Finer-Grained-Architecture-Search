import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from categorical import sample_gumbel, gumbel_softmax_sample, gumbel_softmax
from tensorflow.examples.tutorials.mnist import input_data
import seaborn as sns
from utils import *
from tensorflow.python.tools import inspect_checkpoint as chkp
from IPython import embed
from tensorflow.python import pywrap_tensorflow
import os
import logging
import sys

class GeneralizedConvNetwork(object):
	def __init__(self, config):
		self.config = config
		self.batch_size = self.config["train_batch_size"]
		self.weight_lr = self.config["weight_lr"]
		self.meta_lr = self.config["meta_lr"]
		self.beta = self.config["beta"]
		# If just training hyperparameter (comment out weight training), meta_lr = 0.001, beta = 0.001 seems optimal, can get to 1.00 val accuracy!
		self.build_model()
		self.add_summaries()
		return


	def build_model(self): 
		#Placeholders 
		self.train_input = tf.placeholder(tf.float32, [None, 784])
		self.train_input_reshape = tf.reshape(self.train_input, [-1, 28, 28, 1])
		self.train_label = tf.placeholder(tf.float32, [None, 10]) 
		self.test_input = tf.placeholder(tf.float32, [None, 784])
		self.test_input_reshape = tf.reshape(self.test_input, [-1, 28, 28, 1]) 
		self.test_label = tf.placeholder(tf.float32, [None, 10])

		#Additional Placeholders
		self.train_weights = tf.placeholder(tf.int32, name = "train_weights")
		self.exploration = tf.placeholder(tf.int32, name = "exploration")

		#Weights
		with tf.variable_scope("Weights", reuse = tf.AUTO_REUSE):
			self.generate_weights(self.config["model"])

		self.output = self.feed_forward(self.train_input_reshape, self.weights, self.config["model"])
		self.train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.train_label))

		
		grads = tf.gradients(self.train_loss, list(self.weights.values()))
		self.gradients = dict(zip(self.weights.keys(), grads))
		look_ahead_weights = dict(zip(self.weights.keys(), [self.weights[key] - self.beta*self.gradients[key] for key in self.weights.keys()]))


		self.output1 = self.feed_forward(self.test_input_reshape, look_ahead_weights, self.config["model"])
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
		hyper_grad = self.hyper_optimizer.compute_gradients(self.test_loss, hyper_vars) 
		self.hyper_grad = hyper_grad
		self.hyper_update = self.hyper_optimizer.apply_gradients(hyper_grad)
		
		#Statistics & Miscellaneous
		weight_correct_pred = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.train_label, 1))
		self.weight_accuracy = tf.reduce_mean(tf.cast(weight_correct_pred, tf.float32))

		hyper_correct_pred = tf.equal(tf.argmax(self.output1, 1), tf.argmax(self.test_label, 1))
		self.hyper_accuracy = tf.reduce_mean(tf.cast(hyper_correct_pred, tf.float32))

		self.assign =[]
		self.prev_weight_ph = []
		for counter, var in enumerate(weight_vars):
				self.prev_weight_ph += [tf.placeholder(tf.float32, var.get_shape())]
				self.assign+=[tf.assign(var, self.prev_weight_ph[counter])]

	def generate_weights(self, model_config):
		self.weights= {}
		counter =0
		layer_list = model_config["layers"]
		with tf.variable_scope("Weights", reuse = tf.AUTO_REUSE):
			for i in range(0, len(layer_list)):
				layer_dict = layer_list[i]
				if layer_dict["type"]=="FC":
					self.weights["w" + str(counter)] = tf.get_variable("w" + str(counter), 
						[layer_dict["input"], layer_dict["output"]], 
						initializer = layer_dict["weight_initializer"] if layer_dict["weight_initializer"] else tf.contrib.layers.xavier_initializer())
					self.weights["b" + str(counter)] = tf.get_variable("b" + str(counter), 
						[layer_dict["output"]], 
						initializer = layer_dict["bias_initalizer"] if layer_dict["bias_initalizer"] else tf.zeros_initializer())
					counter+=1
				elif layer_dict["type"]=="Conv" or layer_dict["type"]=="Dense":
					self.weights["w" + str(counter)] = tf.get_variable("w" + str(counter), 
						[layer_dict["filter_height"], layer_dict["filter_width"], layer_dict["in_channels"], layer_dict["out_channels"]], 
						initializer = layer_dict["weight_initializer"] if layer_dict["weight_initializer"] else tf.contrib.layers.xavier_initializer_conv2d())
					self.weights["b" + str(counter)] = tf.get_variable("b" + str(counter), 
						[layer_dict["out_channels"]], 
						initializer = layer_dict["bias_initalizer"] if layer_dict["bias_initalizer"] else tf.zeros_initializer())
					counter+=1

	def add_summaries(self):
		with tf.name_scope('Weight-Training'):
			self.weight_loss_summ = tf.summary.scalar('Weight-Train Loss', self.train_loss)
			self.weight_acc_summ = tf.summary.scalar('Weight Accuracy', self.weight_accuracy)
			self.weight_merged = tf.summary.merge([self.weight_loss_summ, self.weight_acc_summ])

		with tf.name_scope('Hyper-Training'):
			self.hyper_loss_summ = tf.summary.scalar('Hyper-Train Loss', self.test_loss)
			self.hyper_acc_summ = tf.summary.scalar('Hyper Accuracy', self.hyper_accuracy)
			self.hyper_merged = tf.summary.merge([self.hyper_loss_summ, self.hyper_acc_summ])
	
	def feed_forward(self, inp, weights, model_config):
		layer_list = model_config["layers"]
		counter =0
		output = inp
		print(output)
		for i in range(0, len(layer_list)):
			layer_dict = layer_list[i]
			
			activation_fn = get_activation_function(layer_dict["activation"]) if "activation" in layer_dict else None 
			norm_type = layer_dict["norm"] if "norm" in layer_dict else None		

			if layer_dict["type"]=="FC":
				output = FCLayer(output, weights["w"+str(counter)], weights["b"+str(counter)], activation_fn, norm_type)
				counter+=1
			elif layer_dict["type"]=="Conv":
				pool_config = layer_dict["pool"] if "pool" in layer_dict else None
				output = ConvLayer(output, weights["w"+str(counter)], weights["b"+str(counter)], layer_dict["stride"], activation_fn, norm_type, pool_config)
				counter+=1
			elif layer_dict["type"]=="Dense":
				pool_config = layer_dict["pool"] if "pool" in layer_dict else None
				output = DenseLayer(output, layer_dict["output_dim"] , weights["w"+str(counter)], weights["b"+str(counter)], self.train_weights, self.exploration, activation_fn, norm_type, pool_config)
				counter+=1

			print(output)
		return output