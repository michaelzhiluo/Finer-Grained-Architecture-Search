import numpy as np
import tensorflow as tf


def DenseLayer(inputs, 
			output, 
			activation_fn = tf.nn.relu,
			num_weights_per_filter = 25, 
			num_filters = 32,
			weights_initalizer = initializers.xavier_initializer(),
			biases_initializer=tf.zeros_initializer(),
			weight_scope = "Weights",
			hyperparam_scope = "Hyperparameters"):
	'''
	inputs-Input tensor of shape [batch_size, , in_channels]
	'''

	#Flatten input to [batch_size, total_input_dim]
	input_shape = tf.shape(inputs)
	inputs = tf.manip.reshape(inputs, [None, tf.reduce_prod(input_shape[1:])])


	with tf.variable_scope(weight_scope):
		weights = tf.get_variable("weight", shape=[num_filters, num_weights_per_filter, input_shape[2]])

	with tf.variable_scope(hyperparam_scope):
		alpha = tf.get_variable("hyperparam", shape=[])
	
	