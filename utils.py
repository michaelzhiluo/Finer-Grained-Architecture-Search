import numpy as np
import tensorflow as tf
from categorical import sample_gumbel, gumbel_softmax_sample, gumbel_softmax
from tensorflow.examples.tutorials.mnist import input_data

def DenseLayer(inputs, 
			output, 
			conv_weight,
			bias,
			weight_train,
			exploration,
			activation_fn = tf.nn.relu,
			hyperparam_scope = "Hyperparameters"):
	'''
	DenseLayer takes any image, converts it to [batchsize, image_size]. Image size is all the image dimensions multiplied together.
	For example, a 100 10x10x3 standard images will reduce [100, 10*10*3].

	inputs-Input tensor of shape [batch_size] + [Any Image Shape] (it will be flattened anyways)
	output- Scalar Output Size e.g 100
	num_filters - Output filters 
	'''

	#Flatten/vectorizes input to [batch_size, input_size]
	inputs = tf.contrib.layers.flatten(inputs)
	
	with tf.variable_scope(hyperparam_scope, reuse = tf.AUTO_REUSE):
		alpha = tf.get_variable("alpha", shape=[output, conv_weight.get_shape()[1], inputs.get_shape()[1].value])

	#SoftMax Alpha
	s_alpha = tf.nn.softmax(alpha, name="softmax_alpha")

	# If exploration, sample uniformly (infinitely high temperature), elsewise sample from your learned alphas with temperature set to 1
	dist = tf.cond(exploration > 0, lambda: gumbel_softmax(alpha, 100000000000, True), lambda:  gumbel_softmax(alpha, 1, True))

	sampled_connections = tf.einsum("abc,dc->dab", dist, inputs)

	max_connections = tf.gather(inputs, tf.argmax(s_alpha, axis = 2, output_type=tf.int32, name="argmax_alpha"), axis=1)

	connections = tf.cond(weight_train>0, lambda: sampled_connections, lambda: sampled_connections)

	multiplied_output = tf.einsum("aij,bj->aib", connections, conv_weight)

	final_output = activation_fn(tf.nn.bias_add(multiplied_output, bias))

	return final_output
'''
class Explore(object):
	def __init__(self, initial_exp, exp_decay):
		self.initial_exp = initial_exp
		self.exp_decay  = 
		'''